/* Copyright (c) 2017-2018 Mozilla */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kiss_fft.h"
#include "common.h"
#include <math.h>
#include "freq.h"
#include "pitch.h"
#include "arch.h"
#include "celt_lpc.h"
#include <assert.h>


#define PITCH_MIN_PERIOD 32
#define PITCH_MAX_PERIOD 256
#define PITCH_FRAME_SIZE 320
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD+PITCH_FRAME_SIZE)

#define CEPS_MEM 8
#define NB_DELTA_CEPS 6

#define NB_FEATURES (2*NB_BANDS+3+LPC_ORDER)


typedef struct {
  float analysis_mem[OVERLAP_SIZE];
  float cepstral_mem[CEPS_MEM][NB_BANDS];
  float pitch_buf[PITCH_BUF_SIZE];
  float exc_buf[PITCH_BUF_SIZE];
  float last_gain;
  int last_period;
  float lpc[LPC_ORDER];
  float features[4][NB_FEATURES];
  float sig_mem[LPC_ORDER];
  int exc_mem;
} DenoiseState;

static int rnnoise_get_size() {
  return sizeof(DenoiseState);
}

static int rnnoise_init(DenoiseState *st) {
  memset(st, 0, sizeof(*st));
  return 0;
}

static DenoiseState *rnnoise_create() {
  DenoiseState *st;
  st = malloc(rnnoise_get_size());
  rnnoise_init(st);
  return st;
}

static void rnnoise_destroy(DenoiseState *st) {
  free(st);
}

static short float2short(float x)
{
  int i;
  i = (int)floor(.5+x);
  return IMAX(-32767, IMIN(32767, i));
}

int lowpass = FREQ_SIZE;
int band_lp = NB_BANDS;

static void frame_analysis(DenoiseState *st, kiss_fft_cpx *X, float *Ex, const float *in) {
  int i;
  float x[WINDOW_SIZE];
  RNN_COPY(x, st->analysis_mem, OVERLAP_SIZE);
  RNN_COPY(&x[OVERLAP_SIZE], in, FRAME_SIZE);
  RNN_COPY(st->analysis_mem, &in[FRAME_SIZE-OVERLAP_SIZE], OVERLAP_SIZE);
  apply_window(x);
  forward_transform(X, x);
  for (i=lowpass;i<FREQ_SIZE;i++)
    X[i].r = X[i].i = 0;
  compute_band_energy(Ex, X);
}

static void compute_frame_features(DenoiseState *st, FILE *ffeat, const float *in) {
  float aligned_in[FRAME_SIZE];
  static int pcount = 0;
  int i;
  float E = 0;
  float Ly[NB_BANDS];
  float follow, logMax;
  float g;
  kiss_fft_cpx X[FREQ_SIZE];
  float Ex[NB_BANDS];
  RNN_COPY(aligned_in, &st->analysis_mem[OVERLAP_SIZE-TRAINING_OFFSET], TRAINING_OFFSET);
  frame_analysis(st, X, Ex, in);
  logMax = -2;
  follow = -2;
  for (i=0;i<NB_BANDS;i++) {
    Ly[i] = log10(1e-2+Ex[i]);
    Ly[i] = MAX16(logMax-8, MAX16(follow-2.5, Ly[i]));
    logMax = MAX16(logMax, Ly[i]);
    follow = MAX16(follow-2.5, Ly[i]);
    E += Ex[i];
  }
  dct(st->features[pcount], Ly);
  st->features[pcount][0] -= 4;
  g = lpc_from_cepstrum(st->lpc, st->features[pcount]);
  st->features[pcount][2*NB_BANDS+2] = log10(g);
  for (i=0;i<LPC_ORDER;i++) st->features[pcount][2*NB_BANDS+3+i] = st->lpc[i];
  {
    float xcorr[PITCH_MAX_PERIOD];
    static float mem[LPC_ORDER];
    static float filt=0;
    float best_corr = -100;
    int best_period = 2*PITCH_MIN_PERIOD;
    float ener0;
    RNN_MOVE(st->exc_buf, &st->exc_buf[FRAME_SIZE], PITCH_MAX_PERIOD);
    RNN_COPY(&aligned_in[TRAINING_OFFSET], in, FRAME_SIZE-TRAINING_OFFSET);
    for (i=0;i<FRAME_SIZE;i++) {
      int j;
      float sum = aligned_in[i];
      for (j=0;j<LPC_ORDER;j++)
        sum += st->lpc[j]*mem[j];
      RNN_MOVE(mem+1, mem, LPC_ORDER-1);
      mem[0] = aligned_in[i];
      st->exc_buf[PITCH_MAX_PERIOD+i] = sum + .7*filt;
      filt = sum;
      //printf("%f\n", st->exc_buf[PITCH_MAX_PERIOD+i]);
    }
    int sub;
    static float xc[10][PITCH_MAX_PERIOD+1];
    static float ener[10][PITCH_MAX_PERIOD];
    static float frame_max_corr[PITCH_MAX_PERIOD];
    /* Cross-correlation on half-frames. */
    for (sub=0;sub<2;sub++) {
      int off = sub*FRAME_SIZE/2;
      celt_pitch_xcorr(&st->exc_buf[PITCH_MAX_PERIOD+off], st->exc_buf+off, xcorr, FRAME_SIZE/2, PITCH_MAX_PERIOD);
      ener0 = celt_inner_prod(&st->exc_buf[PITCH_MAX_PERIOD+off], &st->exc_buf[PITCH_MAX_PERIOD+off], FRAME_SIZE/2);
      for (i=0;i<PITCH_MAX_PERIOD;i++) {
        ener[2+2*pcount+sub][i] = (1 + ener0 + celt_inner_prod(&st->exc_buf[i+off], &st->exc_buf[i+off], FRAME_SIZE/2));
        xc[2+2*pcount+sub][i] = 2*xcorr[i] / ener[2+2*pcount+sub][i];
      }
#if 0
      for (i=0;i<PITCH_MAX_PERIOD;i++)
        printf("%f ", xc[2*pcount+sub][i]);
      printf("\n");
#endif
    }
    pcount++;
    /* Running on groups of 4 frames. */
    if (pcount == 4) {
      int period;
      float best_a=0;
      float best_b=0;
      float w;
      float sx=0, sxx=0, sxy=0, sy=0, sw=0;
      float sc=0;
      float frame_corr;
      int voiced;
      best_corr = -100;
      best_period = PITCH_MIN_PERIOD;
      /* Search approximate pitch by considering the max correlation over all sub-frames
         within a window corresponding to 25% of the pitch (4 semitones). */
      for (i=PITCH_MAX_PERIOD-PITCH_MIN_PERIOD*5/4;i>=0;i--) {
        int j;
        float corr;
        period = PITCH_MAX_PERIOD - i;
        float num=0;
        float den=0;
        for (sub=0;sub<10;sub++) {
          float max_xc=-1000, max_ener=0;
          for (j=0;j<period/5;j++) {
            if (xc[sub][i+j] > max_xc) {
              max_xc = xc[sub][i+j];
              max_ener = ener[sub][i+j];
            }
          }
          num += max_xc*max_ener;
          den += max_ener;
        }
        corr = num/den;
        corr = MAX16(corr, frame_max_corr[i]-.15);
        frame_max_corr[i] = corr;
        if (corr > best_corr) {
          if (period < best_period*5/4 || (corr > 1.2*best_corr && best_corr < .5)) {
            best_corr = corr;
            best_period = period;
          }
        }
        //printf("%f ", corr);
      }
      int best[10];
      i = PITCH_MAX_PERIOD - best_period;
      period = best_period;
      for (sub=0;sub<10;sub++) {
        int j;
        int sub_period = PITCH_MIN_PERIOD;
        float max_xc=-1000, max_ener=0;
        for (j=0;j<period/5;j++) {
          float curr;
          curr = xc[sub][i+j];
          if (sub > 0 && sub < 9) {
            curr = .5*xc[sub][i+j] + .25*MAX16(MAX16(xc[sub-1][i+j]+xc[sub+1][i+j], xc[sub-1][i+j-1]+xc[sub+1][i+j+1]), xc[sub-1][i+j+1]+xc[sub+1][i+j-1]);
          }
          if (curr > max_xc) {
            max_xc = curr;
            max_ener = ener[sub][i+j];
            sub_period = period - j;
          }
        }
        w = MAX16(1e-1, MIN16(1, 5*(max_xc-.2)))*max_ener;
        w = MAX16(.1, MIN16(1, 5*(max_xc-.2)));
        sw += w;
        sx += w*sub;
        sxx += w*sub*sub;
        sxy += w*sub*sub_period;
        sy += w*sub_period;
        sc += w*max_xc;
        best[sub] = sub_period;
      }
      frame_corr = sc/sw;
      voiced = frame_corr > .45;
      /* Linear regression to figure out the pitch contour. */
      best_a = (sw*sxy - sx*sy)/(sw*sxx - sx*sx);
      if (voiced) {
        float mean_pitch = sy/sw;
        /* Allow a relative variation of up to 1/4 over 8 sub-frames. */
        float max_a = mean_pitch/32;
        best_a = MIN16(max_a, MAX16(-max_a, best_a));
      } else {
        best_a = 0;
      }
      //best_b = (sxx*sy - sx*sxy)/(sw*sxx - sx*sx);
      best_b = (sy - best_a*sx)/sw;
      /* Quantizing the pitch as "main" pitch + slope. */
      float center_pitch = best_b+5.5*best_a;
      int main_pitch = (int)floor(.5 + 21.*log2(center_pitch/PITCH_MIN_PERIOD));
      main_pitch = IMAX(0, IMIN(63, main_pitch));
      int modulation = (int)floor(.5 + 16*7*best_a/center_pitch);
      modulation = IMAX(-3, IMIN(3, modulation));
      //printf("%d %d\n", main_pitch, modulation);
      //printf("%f %f\n", best_a/center_pitch, best_corr);
      //for (sub=2;sub<10;sub++) printf("%f %d %f\n", best_b + sub*best_a, best[sub], best_corr);
      for (sub=0;sub<4;sub++) {
          float p = pow(2.f, main_pitch/21.)*PITCH_MIN_PERIOD;
          p *= 1 + modulation/16./7.*(2*sub-3);
          st->features[sub][2*NB_BANDS] = .02*(p-100);
          st->features[sub][2*NB_BANDS + 1] = voiced ? 1 : -1;
          //printf("%f %f %d %f %f\n", st->features[sub][2*NB_BANDS], p, best[2+2*sub], best_corr, frame_corr);
      }
      //printf("%d %f %f %f\n", best_period, best_a, best_b, best_corr);
      RNN_COPY(&xc[0][0], &xc[8][0], PITCH_MAX_PERIOD);
      RNN_COPY(&xc[1][0], &xc[9][0], PITCH_MAX_PERIOD);
      RNN_COPY(&ener[0][0], &ener[8][0], PITCH_MAX_PERIOD);
      RNN_COPY(&ener[1][0], &ener[9][0], PITCH_MAX_PERIOD);

      for (i=0;i<4;i++) {
        fwrite(st->features[i], sizeof(float), NB_FEATURES, ffeat);
      }
      pcount=0;
    }
  }
}

static void biquad(float *y, float mem[2], const float *x, const float *b, const float *a, int N) {
  int i;
  for (i=0;i<N;i++) {
    float xi, yi;
    xi = x[i];
    yi = x[i] + mem[0];
    mem[0] = mem[1] + (b[0]*(double)xi - a[0]*(double)yi);
    mem[1] = (b[1]*(double)xi - a[1]*(double)yi);
    y[i] = yi;
  }
}

static void preemphasis(float *y, float *mem, const float *x, float coef, int N) {
  int i;
  for (i=0;i<N;i++) {
    float yi;
    yi = x[i] + *mem;
    *mem = -coef*x[i];
    y[i] = yi;
  }
}

static float uni_rand() {
  return rand()/(double)RAND_MAX-.5;
}

static void rand_resp(float *a, float *b) {
  a[0] = .75*uni_rand();
  a[1] = .75*uni_rand();
  b[0] = .75*uni_rand();
  b[1] = .75*uni_rand();
}

void write_audio(DenoiseState *st, const short *pcm, float noise_std, FILE *file) {
  int i;
  unsigned char data[4*FRAME_SIZE];
  for (i=0;i<FRAME_SIZE;i++) {
    int noise;
    float p=0;
    float e;
    int j;
    for (j=0;j<LPC_ORDER;j++) p -= st->lpc[j]*st->sig_mem[j];
    e = lin2ulaw(pcm[i] - p);
    /* Signal. */
    data[4*i] = lin2ulaw(st->sig_mem[0]);
    /* Prediction. */
    data[4*i+1] = lin2ulaw(p);
    /* Excitation in. */
    data[4*i+2] = st->exc_mem;
    /* Excitation out. */
    data[4*i+3] = e;
    /* Simulate error on excitation. */
    noise = (int)floor(.5 + noise_std*.707*(log_approx((float)rand()/RAND_MAX)-log_approx((float)rand()/RAND_MAX)));
    e += noise;
    e = IMIN(255, IMAX(0, e));
    
    RNN_MOVE(&st->sig_mem[1], &st->sig_mem[0], LPC_ORDER-1);
    st->sig_mem[0] = p + ulaw2lin(e);
    st->exc_mem = e;
  }
  fwrite(data, 4*FRAME_SIZE, 1, file);
}

int main(int argc, char **argv) {
  int i;
  int count=0;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  float a_sig[2] = {0};
  float b_sig[2] = {0};
  float mem_hp_x[2]={0};
  float mem_resp_x[2]={0};
  float mem_preemph=0;
  float x[FRAME_SIZE];
  int gain_change_count=0;
  FILE *f1;
  FILE *ffeat;
  FILE *fpcm=NULL;
  short pcm[FRAME_SIZE]={0};
  short tmp[FRAME_SIZE] = {0};
  float savedX[FRAME_SIZE] = {0};
  float speech_gain=1;
  int last_silent = 1;
  float old_speech_gain = 1;
  int one_pass_completed = 0;
  DenoiseState *st;
  float noise_std=0;
  int training = -1;
  st = rnnoise_create();
  if (argc == 5 && strcmp(argv[1], "-train")==0) training = 1;
  if (argc == 4 && strcmp(argv[1], "-test")==0) training = 0;
  if (training == -1) {
    fprintf(stderr, "usage: %s -train <speech> <features out> <pcm out>\n", argv[0]);
    fprintf(stderr, "  or   %s -test <speech> <features out>\n", argv[0]);
    return 1;
  }
  f1 = fopen(argv[2], "r");
  if (f1 == NULL) {
    fprintf(stderr,"Error opening input .s16 16kHz speech input file: %s\n", argv[2]);
    exit(1);
  }
  ffeat = fopen(argv[3], "w");
  if (ffeat == NULL) {
    fprintf(stderr,"Error opening output feature file: %s\n", argv[3]);
    exit(1);
  }
  if (training) {
    fpcm = fopen(argv[4], "w");
    if (fpcm == NULL) {
      fprintf(stderr,"Error opening output PCM file: %s\n", argv[4]);
      exit(1);
    }
  }
  while (1) {
    float E=0;
    int silent;
    for (i=0;i<FRAME_SIZE;i++) x[i] = tmp[i];
    fread(tmp, sizeof(short), FRAME_SIZE, f1);
    if (feof(f1)) {
      if (!training) break;
      rewind(f1);
      fread(tmp, sizeof(short), FRAME_SIZE, f1);
      one_pass_completed = 1;
    }
    for (i=0;i<FRAME_SIZE;i++) E += tmp[i]*(float)tmp[i];
    if (training) {
      silent = E < 5000 || (last_silent && E < 20000);
      if (!last_silent && silent) {
        for (i=0;i<FRAME_SIZE;i++) savedX[i] = x[i];
      }
      if (last_silent && !silent) {
          for (i=0;i<FRAME_SIZE;i++) {
            float f = (float)i/FRAME_SIZE;
            tmp[i] = (int)floor(.5 + f*tmp[i] + (1-f)*savedX[i]);
          }
      }
      if (last_silent) {
        last_silent = silent;
        continue;
      }
      last_silent = silent;
    }
    if (count*FRAME_SIZE_5MS>=10000000 && one_pass_completed) break;
    if (training && ++gain_change_count > 2821) {
      float tmp;
      speech_gain = pow(10., (-20+(rand()%40))/20.);
      if (rand()%20==0) speech_gain *= .01;
      if (rand()%100==0) speech_gain = 0;
      gain_change_count = 0;
      rand_resp(a_sig, b_sig);
      tmp = (float)rand()/RAND_MAX;
      noise_std = 4*tmp*tmp;
    }
    biquad(x, mem_hp_x, x, b_hp, a_hp, FRAME_SIZE);
    biquad(x, mem_resp_x, x, b_sig, a_sig, FRAME_SIZE);
    preemphasis(x, &mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
    for (i=0;i<FRAME_SIZE;i++) {
      float g;
      float f = (float)i/FRAME_SIZE;
      g = f*speech_gain + (1-f)*old_speech_gain;
      x[i] *= g;
    }
    for (i=0;i<FRAME_SIZE;i++) x[i] += rand()/(float)RAND_MAX - .5;
    compute_frame_features(st, ffeat, x);
    /* PCM is delayed by 1/2 frame to make the features centered on the frames. */
    for (i=0;i<FRAME_SIZE-TRAINING_OFFSET;i++) pcm[i+TRAINING_OFFSET] = float2short(x[i]);
    if (fpcm) write_audio(st, pcm, noise_std, fpcm);
    //if (fpcm) fwrite(pcm, sizeof(short), FRAME_SIZE, fpcm);
    for (i=0;i<TRAINING_OFFSET;i++) pcm[i] = float2short(x[i+FRAME_SIZE-TRAINING_OFFSET]);
    old_speech_gain = speech_gain;
    count++;
  }
  fclose(f1);
  fclose(ffeat);
  if (fpcm) fclose(fpcm);
  rnnoise_destroy(st);
  return 0;
}
