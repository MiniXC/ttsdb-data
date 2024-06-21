from pathlib import Path
import random

wavs = sorted(list(Path("parler").rglob("*.wav")))

# seed
random.seed(0)
random.shuffle(wavs)

parler_processed = Path("parler_processed")
parler_processed.mkdir(parents=True, exist_ok=True)

for wav in wavs[:100]:
    wav_name = wav.name
    txt_name = wav_name.replace(".wav", ".txt")
    Path(f"parler_processed/{wav_name}").write_bytes(wav.read_bytes())

    # write the text file
    text = Path(f"parler/{txt_name}").read_text()
    Path(f"parler_processed/{txt_name}").write_text(text)