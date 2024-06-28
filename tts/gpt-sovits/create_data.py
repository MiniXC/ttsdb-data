import sys
from pathlib import Path
import soundfile as sf
import os
import tempfile
import librosa
import numpy as np
from tqdm import tqdm

os.chdir("GPT-SoVITS")
sys.path.append('.')

from inference_webui import get_tts_wav

def create_wav(speaker, text):
    # create a temporary file of the speaker ref that is 8 seconds long
    y, sr = librosa.load(speaker)
    
    # if shorter than 3 seconds, pad the start and end with zeros
    if len(y) < sr * 3:
        y = np.pad(y, (sr * 3 - len(y), 0), mode="constant")
    # if longer than 8 seconds, take a random 8 second clip
    elif len(y) > sr * 8:
        start = np.random.randint(0, len(y) - sr * 8)
        y = y[start:start + sr * 8]

    temp_file = tempfile.mktemp(suffix=".wav")
    sf.write(temp_file, y, sr)

    wav = get_tts_wav(str(temp_file), None, "en", text, "en")

    wavs = []

    for chunk in wav:
        sr, wav = chunk
        wavs.append(wav)

    return sr, np.concatenate(wavs)

if __name__ == "__main__":
    
    Path("../../data/gptsovits").mkdir(parents=True, exist_ok=True)
    wavs = sorted(list(Path("../../data/libritts_test_processed").rglob("*.wav")))
    txts = sorted(list(Path("../../data/libritts_test_processed").rglob("*.txt")))

    output_dir = "../../data/gptsovits"

    speaker_dict = {}

    for wav, txt in tqdm(zip(wavs, txts), total=len(wavs)):
        speaker_id = wav.stem.split("-")[0]

        if speaker_id not in speaker_dict:
            speaker_dict[speaker_id] = wav.resolve()

        speaker = speaker_dict[speaker_id]

        wav_name = f"{speaker_id}-{wav.stem.split('-')[1]}.wav"
        txt_name = f"{speaker_id}-{wav.stem.split('-')[1]}.txt"

        text = txt.read_text()

        sr, wav = create_wav(speaker, text)

        sf.write(f"{output_dir}/{wav_name}", wav, sr)

        # write the text file
        Path(f"{output_dir}/{txt_name}").write_text(text)