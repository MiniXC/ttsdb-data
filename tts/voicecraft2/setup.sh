pip install torchaudio==2.0.2 torch==2.0.1 xformers==0.0.22 # this assumes your system is compatible with CUDA 11.7, otherwise checkout https://pytorch.org/get-started/previous-versions/#v201
pip install -e git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft
apt-get install ffmpeg # if you don't already have ffmpeg installed
apt-get install espeak-ng # backend for the phonemizer installed below
pip install tensorboard==2.16.2
pip install phonemizer==3.2.1
pip install datasets==2.16.0
pip install torchmetrics==0.11.1
# install MFA for getting forced-alignment, this could take a few minutes
conda install -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068
# install MFA english dictionary and model
mfa model download dictionary english_us_arpa
mfa model download acoustic english_us_arpa
pip install huggingface_hub
git clone https://github.com/jasonppy/VoiceCraft.git