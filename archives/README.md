# Reference Datasets

## Blizzard 2008 & 2013
The data can be found here: https://www.cstr.ed.ac.uk/projects/blizzard/data.html
We use "2008 Version 1" and "2013 extension conducted in 2023 Version 1"

## LibriTTS
We use dev-clean.tar.gz and test-clean.tar.gz, which can be found here: https://openslr.org/60/

## LJSpeech
We use a custom test split of the full dataset, which can be found here: https://keithito.com/LJ-Speech-Dataset/

## VCTK
We use version 0.92, which can be found here: https://datashare.ed.ac.uk/handle/10283/3443

## CommonVoice
We use version 1, which can be found here: https://commonvoice.mozilla.org/en/datasets

# Noise Datasets

We use different types of noise to ensure stable scores
- All 0s (Silence)
- All 1s (Constant)
- Normal Distribution
- Uniform Distribution
- Environmental (ESC-50, see: https://github.com/karolpiczak/ESC-50)