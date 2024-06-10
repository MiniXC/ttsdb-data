import tarfile
import zipfile
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import pandas as pd

# speech
blizzard2008 = tarfile.open('archives/blizzard_wavs_and_scores_2008_release_version_1.tar.bz2')
blizzard2013 = tarfile.open('archives/2013-EH2-EXT.tar.gz')
libritts_test = tarfile.open('archives/test-clean.tar.gz')
lj_speech = tarfile.open('archives/LJSpeech-1.1.tar.bz2')
vctk = zipfile.ZipFile('archives/VCTK-Corpus-0.92.zip')
common_voice = tarfile.open('archives/en.tar')

# noise
esc50 = zipfile.ZipFile('archives/ESC-50-master.zip')
# the rest of the noise datasets are generated using numpy

target_dir = Path('extracted_data')

# speech
if not target_dir.exists():
    target_dir.mkdir()

if not (target_dir / 'blizzard2008').exists():
    print('Extracting Blizzard 2008')
    blizzard2008.extractall(target_dir / 'blizzard2008')
if not (target_dir / 'blizzard2013').exists():
    print('Extracting Blizzard 2013')
    blizzard2013.extractall(target_dir / 'blizzard2013')
if not (target_dir / 'libritts_test').exists():
    print('Extracting LibriTTS test')
    libritts_test.extractall(target_dir / 'libritts_test')
if not (target_dir / 'lj_speech').exists():
    print('Extracting LJSpeech')
    lj_speech.extractall(target_dir / 'lj_speech')
if not (target_dir / 'vctk').exists():
    print('Extracting VCTK')
    vctk.extractall(target_dir / 'vctk')
if not (target_dir / 'common_voice').exists():
    print('Extracting Common Voice')
    common_voice.extractall(target_dir / 'common_voice')
if not (target_dir / 'esc50').exists():
    print('Extracting ESC-50')
    esc50.extractall(target_dir / 'esc50')

processed_dir = Path('processed_data')
processed_dir.mkdir(exist_ok=True)

libritts_transcripts = pd.read_csv("extracted_data/libritts_test/LibriTTS/eval_sentences10.tsv", sep='\t', names=["id", "transcript"])
transcripts = libritts_transcripts['transcript'].values
# pick 100 random transcripts
np.random.seed(42)
transcripts = np.random.choice(transcripts, 100, replace=False)
transcripts = [
    str(x).replace('\n', ' ') for x in transcripts
]

# blizzard 08 transcripts
# extracted_data/blizzard2008/blizzard_wavs_and_scores_2008_release_version_1/test_sentences/english/full/2008
# get all files in the directory (they don't have extensions)
blizzard2008_transcripts = {}
for file in (target_dir / 'blizzard2008' / 'blizzard_wavs_and_scores_2008_release_version_1' / 'test_sentences' / 'english' / 'full' / '2008').rglob('*'):
    if file.is_file():
        # read lines
        with open(file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            # the text before the first space is the file name, the text after the first space is the transcript
            file_name, transcript = line.split(' ', 1)
            file_name = file_name.strip()
            transcript = transcript.strip()
            blizzard2008_transcripts[file_name] = transcript

# blizzard 13 transcripts
blizzard2013_transcripts = pd.read_csv('manual_blizzard_13.tsv', sep='\t')

# vctk transcripts
# extracted_data/vctk/txt
vctk_transcripts = {}
for txt in (target_dir / 'vctk' / 'txt').rglob('*.txt'):
    with open(txt, 'r') as f:
        transcript = f.read()
    vctk_transcripts[txt.stem] = transcript

# lj_speech transcripts
# extracted_data/lj_speech/LJSpeech-1.1/metadata.csv
lj_speech_transcripts = pd.read_csv('extracted_data/lj_speech/LJSpeech-1.1/metadata.csv', sep='|', names=['file', 'transcript', 'normalized_transcript'])

# noise
if not (processed_dir / 'noises').exists():
    (processed_dir / 'noises').mkdir()
    # - All 0s (Silence)
    # - All 1s (Constant)
    # - Normal Distribution
    # - Uniform Distribution
    if not (processed_dir / 'noises' / 'all_zeros').exists():
        (processed_dir / 'noises' / 'all_zeros').mkdir()
        print('Generating all_zeros')
        for i in range(100):
            # create wav file
            np.random.seed(i)
            num_seconds_float = np.random.uniform(0.5, 5)
            num_samples = int(num_seconds_float * 16000)
            samples = np.zeros(num_samples)
            sf.write(processed_dir / 'noises' / 'all_zeros' / f'{i}.wav', samples, 16000)
            # create transcript
            with open(processed_dir / 'noises' / 'all_zeros' / f'{i}.txt', 'w') as f:
                # write transcript from libritts
                f.write(transcripts[i])
    if not (processed_dir / 'noises' / 'all_ones').exists():
        (processed_dir / 'noises' / 'all_ones').mkdir()
        print('Generating all_ones')
        for i in range(100):
            # create wav file
            np.random.seed(i)
            num_seconds_float = np.random.uniform(0.5, 5)
            num_samples = int(num_seconds_float * 16000)
            samples = np.ones(num_samples)
            sf.write(processed_dir / 'noises' / 'all_ones' / f'{i}.wav', samples, 16000)
            # create transcript
            with open(processed_dir / 'noises' / 'all_ones' / f'{i}.txt', 'w') as f:
                # write transcript from libritts
                f.write(transcripts[i])
    if not (processed_dir / 'noises' / 'normal_distribution').exists():
        (processed_dir / 'noises' / 'normal_distribution').mkdir()
        print('Generating normal_distribution')
        for i in range(100):
            # create wav file
            np.random.seed(i)
            num_seconds_float = np.random.uniform(0.5, 5)
            num_samples = int(num_seconds_float * 16000)
            samples = np.random.normal(0, 1, num_samples)
            sf.write(processed_dir / 'noises' / 'normal_distribution' / f'{i}.wav', samples, 16000)
            # create transcript
            with open(processed_dir / 'noises' / 'normal_distribution' / f'{i}.txt', 'w') as f:
                # write transcript from libritts
                f.write(transcripts[i])
    if not (processed_dir / 'noises' / 'uniform_distribution').exists():
        (processed_dir / 'noises' / 'uniform_distribution').mkdir()
        print('Generating uniform_distribution')
        for i in range(100):
            # create wav file
            np.random.seed(i)
            num_seconds_float = np.random.uniform(0.5, 5)
            num_samples = int(num_seconds_float * 16000)
            samples = np.random.uniform(-1, 1, num_samples)
            sf.write(processed_dir / 'noises' / 'uniform_distribution' / f'{i}.wav', samples, 16000)
            # create transcript
            with open(processed_dir / 'noises' / 'uniform_distribution' / f'{i}.txt', 'w') as f:
                # write transcript from libritts
                f.write(transcripts[i])

tar_dir = Path('tarred_data')
tar_dir.mkdir(exist_ok=True)

# make tar files for noise
for noise_type in [
    'all_zeros',
    'all_ones',
    'normal_distribution',
    'uniform_distribution'
]:
    if not (tar_dir / f'noise_{noise_type}.tar.gz').exists():
        print(f'Creating tar file for {noise_type}')
        with tarfile.open(tar_dir / f'noise_{noise_type}.tar.gz', 'w:gz') as tar:
            for noise_file in (processed_dir / 'noises' / noise_type).iterdir():
                tar.add(noise_file, arcname=noise_file.name)
                # transcript
                tar.add(noise_file.with_suffix('.txt'), arcname=noise_file.with_suffix('.txt').name)

# make tar files for speech

# blizzard2008
# extracted_data/blizzard2008/blizzard_wavs_and_scores_2008_release_version_1/A/submission_directory/english/full/2008
if not (tar_dir / 'speech_blizzard2008.tar.gz').exists():
    print('Creating tar file for Blizzard 2008')
    # write the transcript to files
    for file_name, transcript in blizzard2008_transcripts.items():
        with open(f'extracted_data/blizzard2008/blizzard_wavs_and_scores_2008_release_version_1/A/submission_directory/english/full/2008/{file_name}.txt', 'w') as f:
            f.write(transcript)
    with tarfile.open(tar_dir / 'speech_blizzard2008.tar.gz', 'w:gz') as tar:
        for wav in Path('extracted_data/blizzard2008/blizzard_wavs_and_scores_2008_release_version_1/A/submission_directory/english/full/2008').rglob('*.wav'):
            # remove directory structure
            tar.add(wav, arcname=wav.name)
            # transcript
            txt_path = Path('extracted_data/blizzard2008/blizzard_wavs_and_scores_2008_release_version_1/A/submission_directory/english/full/2008/') / wav.with_suffix('.txt').name
            tar.add(txt_path, arcname=txt_path.name)

# blizzard2013
# extracted_data/blizzard2013/2013-EH2-EXT/natural16/submission_directory/2013/EH2-EXT/audiobook_sentences
if not (tar_dir / 'speech_blizzard2013.tar.gz').exists():
    print('Creating tar file for Blizzard 2013')
    # write the transcript to files
    for i, row in blizzard2013_transcripts.iterrows():
        audio = row["audio"].replace('.wav', '')
        with open(f'extracted_data/blizzard2013/2013-EH2-EXT/natural16/submission_directory/2013/EH2-EXT/audiobook_sentences/{audio}.txt', 'w') as f:
            f.write(row['text'])
    with tarfile.open(tar_dir / 'speech_blizzard2013.tar.gz', 'w:gz') as tar:
        for wav in Path('extracted_data/blizzard2013/2013-EH2-EXT/natural16/submission_directory/2013/EH2-EXT/audiobook_sentences').rglob('*.wav'):
            # remove directory structure
            tar.add(wav, arcname=wav.name)
            # transcript
            tar.add(wav.with_suffix('.txt'), arcname=wav.with_suffix('.txt').name)

# libritts_test
# extracted_data/libritts_test/LibriTTS/test-clean
if not (tar_dir / 'speech_libritts_test.tar.gz').exists():
    print('Creating tar file for LibriTTS test')
    wavs = []
    for wav in Path('extracted_data/libritts_test/LibriTTS/test-clean').rglob('*.wav'):
        wavs.append(wav)
    # randomly select 100 wavs
    np.random.seed(0)
    wavs = np.random.choice(wavs, 100, replace=False)
    with tarfile.open(tar_dir / 'speech_libritts_test.tar.gz', 'w:gz') as tar:
        for wav in wavs:
            tar.add(wav, arcname=wav.name)
            # transcript
            # rename .normalized.txt to .txt
            os.rename(wav.with_suffix('.normalized.txt'), wav.with_suffix('.txt'))
            tar.add(wav.with_suffix('.txt'), arcname=wav.with_suffix('.txt').name)

# lj_speech
# extracted_data/lj_speech/LJSpeech-1.1/wavs
if not (tar_dir / 'speech_lj_speech.tar.gz').exists():
    print('Creating tar file for LJSpeech')
    wavs = []
    for wav in Path('extracted_data/lj_speech/LJSpeech-1.1/wavs').rglob('*.wav'):
        wavs.append(wav)
    # randomly select 100 wavs
    np.random.seed(0)
    wavs = np.random.choice(wavs, 100, replace=False)
    for i, wav in enumerate(wavs):
        # write transcript
        with open(wav.with_suffix('.txt'), 'w') as f:
            text = lj_speech_transcripts[lj_speech_transcripts["file"] == wav.stem]['transcript'].values[0]
            f.write(text)
    with tarfile.open(tar_dir / 'speech_lj_speech.tar.gz', 'w:gz') as tar:
        for wav in wavs:
            # remove directory structure
            tar.add(wav, arcname=wav.name)
            # transcript
            tar.add(wav.with_suffix('.txt'), arcname=wav.with_suffix('.txt').name)

# vctk
# extracted_data/vctk/wav48_silence_trimmed
if not (tar_dir / 'speech_vctk.tar.gz').exists():
    print('Creating tar file for VCTK')
    flacs = []
    for flac in Path('extracted_data/vctk/wav48_silence_trimmed').rglob('*.flac'):
        flacs.append(flac)
    # randomly select 100 flacs
    np.random.seed(0)
    flacs = np.random.choice(flacs, 100, replace=False)
    wavs = []
    for flac in flacs:
        wav = flac.with_suffix('.wav')
        # convert flac to wav
        samples, sample_rate = sf.read(flac)
        sf.write(wav, samples, sample_rate)
        wavs.append(wav)
        # write transcript
        with open(flac.with_suffix('.txt'), 'w') as f:
            stem = flac.stem.replace('_mic1', '').replace('_mic2', '')
            f.write(vctk_transcripts[stem])
    with tarfile.open(tar_dir / 'speech_vctk.tar.gz', 'w:gz') as tar:
        for wav in wavs:
            tar.add(wav, arcname=wav.name)
            # transcript
            tar.add(wav.with_suffix('.txt'), arcname=wav.with_suffix('.txt').name)

# common_voice
# extracted_data/common_voice/clips
test_set = pd.read_csv('extracted_data/common_voice/test.tsv', sep='\t')
test_mp3s = [
    x + '.mp3' for x in test_set['path']
]
test_transcripts = test_set['sentence'].values
test_mp3s_transcripts = list(zip(test_mp3s, test_transcripts))
# randomly select 100 mp3s
np.random.seed(0)
indizes = np.arange(len(test_mp3s_transcripts))
indizes = np.random.choice(indizes, 100, replace=False)
test_mp3s_transcripts = [test_mp3s_transcripts[i] for i in indizes]
if not (tar_dir / 'speech_common_voice.tar.gz').exists():
    print('Creating tar file for Common Voice')
    for mp3, text in test_mp3s_transcripts:
        wav = mp3.replace('.mp3', '.wav')
        # convert mp3 to wav
        samples, sample_rate = sf.read(f'extracted_data/common_voice/clips/{mp3}')
        sf.write(f'extracted_data/common_voice/clips/{wav}', samples, sample_rate)
        # write transcript
        with open(f'extracted_data/common_voice/clips/{wav.replace(".wav", ".txt")}', 'w') as f:
            f.write(text)
    with tarfile.open(tar_dir / 'speech_common_voice.tar.gz', 'w:gz') as tar:
        for wav, text in test_mp3s_transcripts:
            wav = wav.replace('.mp3', '.wav')
            tar.add(f'extracted_data/common_voice/clips/{wav}', arcname=wav)
            # transcript
            tar.add(f'extracted_data/common_voice/clips/{wav.replace(".wav", ".txt")}', arcname=wav.replace(".wav", ".txt"))

# esc50
# extracted_data/esc50/ESC-50-master/audio
if not (tar_dir / 'esc50.tar.gz').exists():
    print('Creating tar file for ESC-50')
    wavs = []
    for wav in Path('extracted_data/esc50/ESC-50-master/audio').rglob('*.wav'):
        wavs.append(wav)
    # randomly select 100 wavs
    np.random.seed(0)
    wavs = np.random.choice(wavs, 100, replace=False)
    # write transcripts from libritts
    for i, wav in enumerate(wavs):
        with open(wav.with_suffix('.txt'), 'w') as f:
            f.write(transcripts[i])
    with tarfile.open(tar_dir / 'esc50.tar.gz', 'w:gz') as tar:
        for wav in wavs:
            tar.add(wav, arcname=wav.name)
            # transcript
            tar.add(wav.with_suffix('.txt'), arcname=wav.with_suffix('.txt').name)
