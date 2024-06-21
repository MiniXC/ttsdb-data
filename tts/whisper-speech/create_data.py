from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn.functional as F
from whisperspeech.pipeline import Pipeline
pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-small-en+pl.model')

# pipe.generate_to_file("Test sentence", "somefile.wav", lang='en', cps=10.5, speaker='https://upload.wikimedia.org/wikipedia/commons/7/75/Winston_Churchill_-_Be_Ye_Men_of_Valour.ogg')

if __name__ == "__main__":
    
    Path("../data/whisperspeech").mkdir(parents=True, exist_ok=True)
    wavs = sorted(list(Path("../data/libritts_test_processed").rglob("*.wav")))
    txts = sorted(list(Path("../data/libritts_test_processed").rglob("*.txt")))

    output_dir = "../data/whisperspeech"

    for wav, txt in tqdm(zip(wavs, txts), total=len(wavs)):
        speaker_id = wav.stem.split("-")[0]
        
        speaker = f"../data/tmp_speakers/{speaker_id}.wav"

        wav_name = f"{speaker_id}-{wav.stem.split('-')[1]}.wav"
        txt_name = f"{speaker_id}-{wav.stem.split('-')[1]}.txt"

        text = txt.read_text()

        pipe.generate_to_file(
            f"{output_dir}/{wav_name}",
            text=text,
            lang='en',
            cps=10.5,
            speaker=speaker
        )

        # write the text file
        Path(f"{output_dir}/{txt_name}").write_text(text)