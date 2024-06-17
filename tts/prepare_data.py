import tarfile
from pathlib import Path
import random

# extract ../archives/test-clean.tar.gz to tts/data/libritts_test
if not Path("data/libritts_test").exists():
    with tarfile.open("../archives/test-clean.tar.gz", "r:gz") as tar:
        Path("data/libritts_test").mkdir(parents=True, exist_ok=True)
        tar.extractall("data/libritts_test")

# tts/data/libritts_test/LibriTTS/test-clean

Path("data/libritts_test_processed").mkdir(parents=True, exist_ok=True)
# here we will save the processed data, which are 100 random samples from the test-clean dataset
# however, we shuffle the text files to make sure that audio and text are not aligned

wavs = sorted(list(Path("data/libritts_test/LibriTTS/test-clean").rglob("*.wav")))
txts = sorted(
    list(Path("data/libritts_test/LibriTTS/test-clean").rglob("*.normalized.txt"))
)

# seed
random.seed(0)
random.shuffle(wavs)
random.shuffle(txts)

for i, (wav, txt) in enumerate(zip(wavs[:100], txts[:100])):
    wav_name = f"{i:03d}.wav"
    txt_name = f"{i:03d}.txt"
    Path(f"data/libritts_test_processed/{wav_name}").write_bytes(wav.read_bytes())
    Path(f"data/libritts_test_processed/{txt_name}").write_text(txt.read_text())
