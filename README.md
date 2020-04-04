# Convolutional Speech Recognition Implementation

This is a simple implementation of the paper [STATE-OF-THE-ART SPEECH RECOGNITION USING MULTI-STREAM SELF-ATTENTION WITH DILATED 1D CONVOLUTIONS](https://arxiv.org/pdf/1910.00716v1.pdf)

As I searched for a good implementation I didn't find any (paper is for 2019 so it's quit new) so I decided to make a very simple implementation inspired by the paper. Some details like factorizations (both in colvolutional and feed-forward weights) are not included here. Maybe in the future I will add them. I'll also be very happy to accept to contributions to make this project more complete.



## Setup
Install all needed dependecies by:
`pip3 install -r requirements.txt`



## Train
Go to `train.py` and change `path` variable to your dataset path. Your dataset directory should contain a csv file called `dataset.csv` which has 2 columns : audio_path, senetence.

Also this script reads your alphabet from a file named `alphabet.pkl` in the root directory. And finally yoo need a folder named `weights` to save model checkpoints.

finally run:
`python3 train.py`
