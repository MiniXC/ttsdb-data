wget https://huggingface.co/ShoukanLabs/Vokan/resolve/main/Model/epoch_2nd_00012.pth?download=true
wget https://huggingface.co/ShoukanLabs/Vokan/raw/main/Model/config.yml
mkdir -p StyleTTS2/Models/Vokan
mv epoch_2nd_00012.pth?download=true tts/styletts2/StyleTTS2/Models/Vokan/epoch_2nd_00012.pth
mv config.yml StyleTTS2/Models/Vokan/config.yml