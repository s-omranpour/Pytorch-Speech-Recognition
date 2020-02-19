# ConvolutionalSpeechRecognition
This project is inspired from Wave2Letter paper and this repo: https://github.com/LearnedVector/Wav2Letter


todos:
 - improving model architecture (using res blocks, see original wave2letter paper, also see https://github.com/silversparro/wav2letter.pytorch/blob/master/model.py)
 - implement ctcbeamsearch decoder in the decoder file (testing the performance afterwards)
 - adding another preprocessing method (like stft) and testing the performance
 - comparing performance of different preprocessing methods (raw, mfcc, stft, ...) keeping model arch fixded
 - using normalization (zero mean, one std) in the preprocessing and compare with no normalization in performance. 
 - adding WER and CER metrics in the metrics.py file
