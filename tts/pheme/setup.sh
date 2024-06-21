git clone https://github.com/PolyAI-LDN/pheme.git
cd pheme
pip install torch torchvision torchaudio
pip install -r requirements.txt --no-deps
st_dir="ckpt/speechtokenizer/"
mkdir -p ${st_dir}
cd ${st_dir}
wget "https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/SpeechTokenizer.pt"
wget "https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/config.json" 
cd ..
wget "https://huggingface.co/fnlp/USLM/resolve/main/USLM_libritts/unique_text_tokens.k2symbols" 
git clone https://huggingface.co/PolyAI/pheme_small ckpt/pheme
mkdir -p "ckpt/t2s"
mkdir -p "ckpt/s2a"
mv ckpt/pheme/config_t2s.json ckpt/t2s/config.json
mv ckpt/pheme/generation_config.json ckpt/t2s/generation_config.json
mv ckpt/pheme/t2s.bin ckpt/t2s/pytorch_model.bin
mv ckpt/pheme/config_s2a.json ckpt/s2a/config.json
mv ckpt/pheme/s2a.ckpt ckpt/s2a/s2a.ckpt