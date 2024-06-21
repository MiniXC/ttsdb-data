# from https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS_Inference.ipynb
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
sudo apt-get install -y --no-install-recommends tzdata ffmpeg libsox-dev parallel aria2 git git-lfs && git lfs install
pip install -r requirements.txt
mkdir -p GPT_SoVITS/pretrained_models
mkdir -p GPT-SoVITS/tools/damo_asr/models
mkdir -p tools/uvr5
cd GPT_SoVITS/pretrained_models
git clone https://huggingface.co/lj1995/GPT-SoVITS hf_gpt_sovits
cd tools/damo_asr/models
git clone https://www.modelscope.cn/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git
git clone https://www.modelscope.cn/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch.git
git clone https://www.modelscope.cn/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch.git
cd tools/uvr5
git clone https://huggingface.co/Delik/uvr5_weights
git config core.sparseCheckout true
mv hf_gpt_sovits/pretrained_models/GPT-SoVITS/* GPT_SoVITS/pretrained_models/