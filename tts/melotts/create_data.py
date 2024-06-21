import sys
from pathlib import Path

from tqdm import tqdm
import numpy as np

sys.path.append('MeloTTS')

from melo.api import TTS

# Speed is adjustable
speed = 1.0

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = 'cpu'

# English 
text = "Did you ever hear a folk tale about a giant turtle?"
model = TTS(language='EN', device=device)
speaker_ids = model.hps.data.spk2id.values()

if __name__ == "__main__":
    speaker_dict = {}
    np.random.seed(0)
    if not Path("../data/melotts").exists():
        Path("../data/melotts").mkdir(parents=True, exist_ok=True)
    for wav in tqdm(sorted(list(Path("../data/libritts_test_processed").rglob("*.txt"))), desc="Synthesizing"):
        # pick a random speaker from speaker_ids
        speaker_id = np.random.choice(list(speaker_ids))
        target_wav = Path(f"../data/melotts/{wav.stem}.wav")
        text = wav.read_text()
        print(speaker_id)
        model.tts_to_file(text, speaker_id, target_wav, speed=speed)
        Path(f"../data/melotts/{wav.stem}.txt").write_text(text)