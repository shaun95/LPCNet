# Place in 16k-LP7 from TSPSpeech.iso and run to concatenate wave files
# into one headerless training file
for i in /data/dataset/mandarin_female/wavs/*.wav
do
sox $i -r 16000 -c 1 -t sw - > mandarin_female/wavs/${i##*/}.s16 
done
