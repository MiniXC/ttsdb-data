import os
import sys
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from pathlib import Path

from tqdm import tqdm
from melo.api import TTS

os.chdir("OpenVoice")
sys.path.append(".")

ckpt_converter = 'checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

language = "EN_NEWEST"

source_se = torch.load(f'checkpoints_v2/base_speakers/ses/en-newest.pth', map_location=device)


if __name__ == "__main__":
    model = TTS(language=language, device=device)
    speaker_id_ = 0

    Path("../../data/openvoicev2").mkdir(parents=True, exist_ok=True)
    wavs = sorted(list(Path("../../data/libritts_test_processed").rglob("*.wav")))
    txts = sorted(list(Path("../../data/libritts_test_processed").rglob("*.txt")))

    output_dir = "../../data/openvoicev2"

    for wav, txt in tqdm(zip(wavs, txts), total=len(wavs)):
        speaker_id = wav.stem.split("-")[0]
        

        speaker = f"../../data/tmp_speakers/{speaker_id}.wav"
        

        target_se, audio_name = se_extractor.get_se(speaker, tone_color_converter, vad=False)

        wav_name = f"{speaker_id}-{wav.stem.split('-')[1]}.wav"
        txt_name = f"{speaker_id}-{wav.stem.split('-')[1]}.txt"

        text = txt.read_text()
        
        src_path = f'{output_dir}/tmp.wav'
        


        model.tts_to_file(text, speaker_id_, src_path, speed=1.0)
        save_path = f'{output_dir}/{wav_name}'

        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
            message=encode_message)

        # write the text file
        Path(f"../../data/openvoicev2/{txt_name}").write_text(text)