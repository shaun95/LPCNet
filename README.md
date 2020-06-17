# LPCNet



Low complexity implementation of the WaveRNN-based LPCNet algorithm, as described in:

J.-M. Valin, J. Skoglund, [LPCNet: Improving Neural Speech Synthesis Through Linear Prediction](https://jmvalin.ca/papers/lpcnet_icassp2019.pdf), *Submitted for ICASSP 2019*, arXiv:1810.11846.

# About Performance when bridging LPCNet and Acoustic Model like Tacotorn and DeepVoice
Tacotron2 or DeepVoice3 which predictes parameters including ceptral coefficients and 2 pitch parameters undiscriminating is probably not good enough to estimate pitch parameters. However, LPCNet is sensitive to the estimation of pitch coefficents, so it is not recommended to predict the features when training acotron2 or DeepVoice3 replacing mel spectrum with ceptral coefficients and 2 pitch parameters directly.    
Update:
Training your tacotron2 or DeepVoice3 very well may help to estimate the pitch parameters.

# Introduction

Work in progress software for researching low CPU complexity algorithms for speech synthesis and compression by applying Linear Prediction techniques to WaveRNN. High quality speech can be synthesised on regular CPUs (around 3 GFLOP) with SIMD support (AVX, AVX2/FMA, NEON currently supported).

The BSD licensed software is written in C and Python/Keras. For training, a GTX 1080 Ti or better is recommended.

This software is an open source starting point for WaveRNN-based speech synthesis and coding.

__NOTE__: The repo aims to work with Tacotron2. You should figure out the structure of the repo because Menglin has reconstruct the project with CMake to add some patches.

# Quickstart  
## LPCNet  
* Download Speech Material for Training LPCNet    
Suitable training material can be obtained from the [McGill University Telecommunications & Signal Processing Laboratory](http://www-mmsp.ece.mcgill.ca/Documents/Data/). Download the ISO and extract the 16k-LP7 directory, the tools/concat.sh script can be used to generate a headerless file of training samples.
```bash
cd 16k-LP7
sh tools/concat.sh
```
* Generate the tool to process the data    
We preprocess raw data of the audios with the tool `dump_data` we provides. You can get it in `bin` with the following commands.  
```bash
mkdir build
cd build
cmake ..
make -j4
```
* Generate training data    
```bash
cd ../bin
./dump_data -train input.s16 features.f32 data.u8
```
where the first file contains 16 kHz 16-bit raw PCM audio (no header) and the other files are output files. This program makes several passes over the data with different filters to generate a large amount of training data. 
* Train the LPCNet.   
```bash
./train_lpcnet.py features.f32 data.u8
``` 
and it will generate a lpcnet*.h5 file for each iteration. If it stops with a 
   "Failed to allocate RNN reserve space" message try reducing the *batch\_size* variable in train_lpcnet.py.
   
* Test the trained LPCNet    
1). You can synthesise speech with Python and your GPU card:
   ```
   ./dump_data -test test_input.s16 test_features.f32
   ./test_lpcnet.py lpcnet test_features.f32 test.s16
   ```
   Note the .h5 is hard coded in test_lpcnet.py, modify for your .h file.  
2). Or with C on a CPU:
   First extract the model files nnet_data.h and nnet_data.c
   ```
   ./dump_lpcnet.py lpcnet15_384_10_G16_64.h5
   ```
   Then you should import model data to make the C synthesiser, You can get it in `bin` with the following commands:
   ```bash
   cd ../build
   cmake ../ -DWITH_MODEL_DATA=TRUE
   make
   cd ../bin
   ```
   At last, you shuould generate the features, try synthesising from a test feature file and add the audio header:
   ```bash
   ./dump_data -test test_input.s16 test_features.f32
   ./test_lpcnet -lpcnet test_features.f32 test.s16
   ffmpeg -f s16le -ar 16k -ac 1 -i test.s16 test-out.wav
   ```
## Tacotron2
Although the model has 55 dims features when training LPCNet, there are 20 features to be used as input features when inferring the audio. You should enble TACOTRON2 Macro in Makefile to get the features for Training Tacotron2. 
* Generate training data    
You also should generate indepent features for every audio when training Tacotron2 other than concatate all features into one file when training LPCNet.
```bash
#preprocessing
./tools/header_removal.sh
./tools/feature_extract.sh
```
* Hack the Tacotron2 code    
Convert the data generated at the last step which has .f32 extension to what could be loaded with numpy. I merge it to the Tacotron feeder [here](https://github.com/Rayhane-mamah/Tacotron-2/blob/master/tacotron/feeder.py#L192) and [here](https://github.com/Rayhane-mamah/Tacotron-2/blob/master/tacotron/feeder.py#L128) with the following code.
```python
mel_target = np.fromfile(os.path.join(self._mel_dir, meta[0]), dtype='float32')
mel_target = np.resize(mel_target, (-1, self._hparams.num_mels))
```
* Train Tacotron2    
Train the tacotron2 with the features  and follow the steps of [Tacatron2 repo](https://github.com/Rayhane-mamah/Tacotron-2). You need to modify the dimensions of input features(num_mels) to 20.

* Test the trained Tacotron2    
Follow the Tacotron2 repo to get the result. and you get the 20 dims acoustic features which are stored with Numpy format.

## Bridge the Tacotron2 and LPCNet
There are also two ways to synthesis the audio from acoustic features stored with the numpy.    
1). You can synthesise speech with Python and your GPU card. However, the original code only accept the 55 dims acoustic features. The 55 dims features consists with 18 bark scale ceptral coefficients, 2 dims pitch parameters(gain and period), 18 dims Bark-Frequency Energy which could be set to zero, 1 dim logistic LPC residual which is not used and 16 dims lpc. The last 16 dims info could be calculated by the former 18 dims coefficients. So we should add the patch to implement it.     
   Menglin choose the `pybind11` to bridge the python code and C code.    
   At first, we should download `pybind11` and build it in work directory. Then build patch.    
   ```
   cd build	 
   cmake .. -DWITH_PYTHON_PATCH=TRUE
   make ..
   ```
   Menglin has hack the patch in python code by using the hint argument `acoustic`. Use the python synthesiser and add the audio header.
   ```
   ./test_lpcnet.py acoustic f32_for_lptnet.f32 test.s16
   ffmpeg -f s16le -ar 16k -ac 1 -i test.s16 test-out.wav
   ```
   Note the .h5 is hard coded in test_lpcnet.py, modify for your .h file.    
2). Convert the acoustic features stored with the numpy to binary format which the LPCNet C synthesiser used.  
```python
   import numpy as np
   npy_data = np.load("npy_from_tacotron2.npy")
   npy_data = npy_data.reshape((-1,))
   npy_data.tofile("f32_for_lptnet.f32")
```
Then use the C synthesiser and add the audio header:
```bash
   ./test_lpcnet -lpcnet f32_for_lptnet.f32 test.s16
   ffmpeg -f s16le -ar 16k -ac 1 -i test.s16 test-out.wav
```

# Reading Further

1. [LPCNet: DSP-Boosted Neural Speech Synthesis](https://people.xiph.org/~jm/demo/lpcnet/)
2. Sample model files:
https://jmvalin.ca/misc_stuff/lpcnet_models/
