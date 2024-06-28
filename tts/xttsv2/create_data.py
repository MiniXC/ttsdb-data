from pathlib import Path
import soundfile as sf
from tqdm import tqdm

from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

if __name__ == "__main__":
    speaker_dict = {}

    Path("../data/xttsv2").mkdir(parents=True, exist_ok=True)
    wavs = sorted(list(Path("../data/libritts_test_processed").rglob("*.wav")))
    txts = sorted(list(Path("../data/libritts_test_processed").rglob("*.txt")))

    for wav, txt in tqdm(zip(wavs, txts), total=len(wavs)):
        speaker_id = wav.stem.split("-")[0]
        if speaker_id not in speaker_dict:
            speaker_dict[speaker_id] = wav.resolve()
        
        wav_name = f"{speaker_id}-{wav.stem.split('-')[1]}.wav"
        txt_name = f"{speaker_id}-{wav.stem.split('-')[1]}.txt"

        text = txt.read_text()
        
        tts.tts_to_file(
            text=text,
            file_path=f"../data/xttsv2/{wav_name}",
            speaker_wav=speaker_dict[speaker_id],
            language="en"
        )

        # write the text file
        Path(f"../data/xttsv2/{txt_name}").write_text(text)