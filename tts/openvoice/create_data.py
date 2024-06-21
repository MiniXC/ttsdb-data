import os
import sys
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from pathlib import Path

from tqdm import tqdm

os.chdir("OpenVoice")
sys.path.append(".")

ckpt_base = 'checkpoints/base_speakers/EN'
ckpt_converter = 'checkpoints/converter'
device="cuda:0" if torch.cuda.is_available() else "cpu"

base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')




if __name__ == "__main__":
    
    Path("../../data/openvoice").mkdir(parents=True, exist_ok=True)
    wavs = sorted(list(Path("../../data/libritts_test_processed").rglob("*.wav")))
    txts = sorted(list(Path("../../data/libritts_test_processed").rglob("*.txt")))

    output_dir = "../../data/openvoice"

    for wav, txt in tqdm(zip(wavs, txts), total=len(wavs)):
        speaker_id = wav.stem.split("-")[0]
        

        speaker = f"../../data/tmp_speakers/{speaker_id}.wav"

        source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)
        target_se, audio_name = se_extractor.get_se(
            speaker,
            tone_color_converter,
            target_dir='processed',
            vad=True
        )

        wav_name = f"{speaker_id}-{wav.stem.split('-')[1]}.wav"
        txt_name = f"{speaker_id}-{wav.stem.split('-')[1]}.txt"

        text = txt.read_text()
        
        src_path = f'{output_dir}/tmp.wav'
        base_speaker_tts.tts(text, src_path, speaker='default', language='English', speed=1.0)
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=f'{output_dir}/{wav_name}',
            message=encode_message)

        # write the text file
        Path(f"../../data/openvoice/{txt_name}").write_text(text)