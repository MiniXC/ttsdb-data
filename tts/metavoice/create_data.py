import os
import sys
import shutil
import tempfile
import time
from pathlib import Path
from typing import Literal, Optional

import librosa
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm
import librosa
import soundfile as sf

os.chdir("metavoice-src")
sys.path.append(".")

from fam.llm.adapters import FlattenedInterleavedEncodec2Codebook
from fam.llm.decoders import EncodecDecoder
from fam.llm.fast_inference_utils import build_model, main
from fam.llm.inference import (
    EncodecDecoder,
    InferenceConfig,
    Model,
    TiltedEncodec,
    TrainedBPETokeniser,
    get_cached_embedding,
    get_cached_file,
    get_enhancer,
)
from fam.llm.utils import (
    check_audio_file,
    get_default_dtype,
    get_device,
    normalize_text,
)
from fam.telemetry import TelemetryEvent
from fam.telemetry.posthog import PosthogClient

posthog = PosthogClient()  # see fam/telemetry/README.md for more information


class TTS:
    END_OF_AUDIO_TOKEN = 1024

    def __init__(
        self,
        model_name: str = "metavoiceio/metavoice-1B-v0.1",
        *,
        seed: int = 1337,
        output_dir: str = "outputs",
        quantisation_mode: Optional[Literal["int4", "int8"]] = None,
        first_stage_path: Optional[str] = None,
        telemetry_origin: Optional[str] = None,
    ):
        """
        Initialise the TTS model.

        Args:
            model_name: refers to the model identifier from the Hugging Face Model Hub (https://huggingface.co/metavoiceio)
            seed: random seed for reproducibility
            output_dir: directory to save output files
            quantisation_mode: quantisation mode for first-stage LLM.
                Options:
                - None for no quantisation (bf16 or fp16 based on device),
                - int4 for int4 weight-only quantisation,
                - int8 for int8 weight-only quantisation.
            first_stage_path: path to first-stage LLM checkpoint. If provided, this will override the one grabbed from Hugging Face via `model_name`.
            telemetry_origin: A string identifier that specifies the origin of the telemetry data sent to PostHog.
        """

        # NOTE: this needs to come first so that we don't change global state when we want to use
        # the torch.compiled-model.
        self._dtype = get_default_dtype()
        self._device = get_device()
        self._model_dir = snapshot_download(repo_id=model_name)
        self.first_stage_adapter = FlattenedInterleavedEncodec2Codebook(end_of_audio_token=self.END_OF_AUDIO_TOKEN)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        if first_stage_path:
            print(f"Overriding first stage checkpoint via provided model: {first_stage_path}")
        self._first_stage_ckpt = first_stage_path or f"{self._model_dir}/first_stage.pt"

        second_stage_ckpt_path = f"{self._model_dir}/second_stage.pt"
        config_second_stage = InferenceConfig(
            ckpt_path=second_stage_ckpt_path,
            num_samples=1,
            seed=seed,
            device=self._device,
            dtype=self._dtype,
            compile=False,
            init_from="resume",
            output_dir=self.output_dir,
        )
        data_adapter_second_stage = TiltedEncodec(end_of_audio_token=self.END_OF_AUDIO_TOKEN)
        self.llm_second_stage = Model(
            config_second_stage, TrainedBPETokeniser, EncodecDecoder, data_adapter_fn=data_adapter_second_stage.decode
        )
        self.enhancer = get_enhancer("df")

        self.precision = {"float16": torch.float16, "bfloat16": torch.bfloat16}[self._dtype]
        self.model, self.tokenizer, self.smodel, self.model_size = build_model(
            precision=self.precision,
            checkpoint_path=Path(self._first_stage_ckpt),
            spk_emb_ckpt_path=Path(f"{self._model_dir}/speaker_encoder.pt"),
            device=self._device,
            compile=False,
            compile_prefill=True,
            quantisation_mode=quantisation_mode,
        )
        self._seed = seed
        self._quantisation_mode = quantisation_mode
        self._model_name = model_name
        self._telemetry_origin = telemetry_origin

    def synthesise(self, text: str, spk_ref_path: str, top_p=0.95, guidance_scale=3.0, temperature=1.0) -> str:
        """
        text: Text to speak
        spk_ref_path: Path to speaker reference file. Min. 30s of audio required. Supports both local paths & public URIs. Audio formats: wav, flac & mp3
        top_p: Top p for sampling applied to first-stage model. Range [0.9, 1.0] are good. This is a measure of speech stability - improves text following for a challenging speaker
        guidance_scale: Guidance scale [1.0, 3.0] for sampling. This is a measure of speaker similarity - how closely to match speaker identity and speech style.
        temperature: Temperature for sampling applied to both LLMs (first & second stage)

        returns: path to speech .wav file
        """
        text = normalize_text(text)
        spk_ref_path = get_cached_file(spk_ref_path)
        check_audio_file(spk_ref_path)
        spk_emb = get_cached_embedding(
            spk_ref_path,
            self.smodel,
        ).to(device=self._device, dtype=self.precision)

        start = time.time()
        # first stage LLM
        tokens = main(
            model=self.model,
            tokenizer=self.tokenizer,
            model_size=self.model_size,
            prompt=text,
            spk_emb=spk_emb,
            top_p=torch.tensor(top_p, device=self._device, dtype=self.precision),
            guidance_scale=torch.tensor(guidance_scale, device=self._device, dtype=self.precision),
            temperature=torch.tensor(temperature, device=self._device, dtype=self.precision),
        )
        _, extracted_audio_ids = self.first_stage_adapter.decode([tokens])

        b_speaker_embs = spk_emb.unsqueeze(0)

        # second stage LLM + multi-band diffusion model
        wav_files = self.llm_second_stage(
            texts=[text],
            encodec_tokens=[torch.tensor(extracted_audio_ids, dtype=torch.int32, device=self._device).unsqueeze(0)],
            speaker_embs=b_speaker_embs,
            batch_size=1,
            guidance_scale=None,
            top_p=None,
            top_k=200,
            temperature=1.0,
            max_new_tokens=None,
        )

        # enhance using deepfilternet
        wav_file = wav_files[0]
        with tempfile.NamedTemporaryFile(suffix=".wav") as enhanced_tmp:
            self.enhancer(str(wav_file) + ".wav", enhanced_tmp.name)
            shutil.copy2(enhanced_tmp.name, str(wav_file) + ".wav")
            print(f"\nSaved audio to {wav_file}.wav")

        # calculating real-time factor (RTF)
        time_to_synth_s = time.time() - start
        audio, sr = librosa.load(str(wav_file) + ".wav")
        duration_s = librosa.get_duration(y=audio, sr=sr)
        real_time_factor = time_to_synth_s / duration_s
        print(f"\nTotal time to synth (s): {time_to_synth_s}")
        print(f"Real-time factor: {real_time_factor:.2f}")

        posthog.capture(
            TelemetryEvent(
                name="user_ran_tts",
                properties={
                    "model_name": self._model_name,
                    "text": text,
                    "temperature": temperature,
                    "guidance_scale": guidance_scale,
                    "top_p": top_p,
                    "spk_ref_path": spk_ref_path,
                    "speech_duration_s": duration_s,
                    "time_to_synth_s": time_to_synth_s,
                    "real_time_factor": round(real_time_factor, 2),
                    "quantisation_mode": self._quantisation_mode,
                    "seed": self._seed,
                    "first_stage_ckpt": self._first_stage_ckpt,
                    "gpu": torch.cuda.get_device_name(0),
                    "telemetry_origin": self._telemetry_origin,
                },
            )
        )

        return str(wav_file) + ".wav"
    
#     speaker_dict = {}

#     Path("../data/xttsv2").mkdir(parents=True, exist_ok=True)
#     wavs = sorted(list(Path("../data/libritts_test_processed").rglob("*.wav")))
#     txts = sorted(list(Path("../data/libritts_test_processed").rglob("*.txt")))

#     for wav, txt in tqdm(zip(wavs, txts), total=len(wavs)):
#         speaker_id = wav.stem.split("-")[0]
#         if speaker_id not in speaker_dict:
#             speaker_dict[speaker_id] = wav.resolve()
        
#         wav_name = f"{speaker_id}-{wav.stem.split('-')[1]}.wav"
#         txt_name = f"{speaker_id}-{wav.stem.split('-')[1]}.txt"

#         text = txt.read_text()
        
#         tts.tts_to_file(
#             text=text,
#             file_path=f"../data/xttsv2/{wav_name}",
#             speaker_wav=speaker_dict[speaker_id],
#             language="en"
#         )

#         # write the text file
#         Path(f"../data/xttsv2/{txt_name}").write_text(text)

if __name__ == "__main__":
    speaker_dict = {}

    tts = TTS("metavoiceio/metavoice-1B-v0.1")
    
    Path("../../data/metavoice").mkdir(parents=True, exist_ok=True)
    wavs = sorted(list(Path("../../data/libritts_test_processed").rglob("*.wav")))
    txts = sorted(list(Path("../../data/libritts_test_processed").rglob("*.txt")))

    for wav, txt in tqdm(zip(wavs, txts), total=len(wavs)):
        speaker_id = wav.stem.split("-")[0]
        first_wav = False
        if speaker_id not in speaker_dict:
            speaker_dict[speaker_id] = wav.resolve()
            first_wav = True
        speaker_wav, sr = librosa.load(speaker_dict[speaker_id])
        if not first_wav and len(speaker_wav) < 30 * sr:
            speaker_wav, sr = librosa.load(wav)
            # append the next reference audio
            old_wav, sr = librosa.load(speaker_dict[speaker_id])
            old_wav = old_wav.tolist()
            old_wav += [0.0 for _ in range(sr//4)]
            old_wav += speaker_wav.tolist()
            # write to temp file
            temp_file = Path(f"../../data/metavoice/tmp/{speaker_id}.wav")
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            sf.write(temp_file, old_wav, sr)
            speaker_dict[speaker_id] = temp_file

    for wav, txt in tqdm(zip(wavs, txts), total=len(wavs)):
        speaker_id = wav.stem.split("-")[0]

        wav_name = f"{speaker_id}-{wav.stem.split('-')[1]}.wav"
        txt_name = f"{speaker_id}-{wav.stem.split('-')[1]}.txt"

        text = txt.read_text()

        speaker_wav, sr = librosa.load(speaker_dict[speaker_id])
        if len(speaker_wav) < 30 * sr:
            # pad the audio
            print(f'Padded audio for speaker {speaker_id}, length: {len(speaker_wav)/sr}')
            # repeat the audio until it reaches 30s
            speaker_wav = speaker_wav.tolist()
            while len(speaker_wav) <= 30 * sr:
                speaker_wav += speaker_wav
            # write to temp file
            temp_file = Path(f"../../data/metavoice/tmp/{speaker_id}.wav")
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            sf.write(temp_file, speaker_wav, sr)
            speaker_dict[speaker_id] = temp_file

        wav_file = Path(tts.synthesise(
            text=text,
            spk_ref_path=str(speaker_dict[speaker_id]),
            top_p=0.95,
            guidance_scale=3.0,
            temperature=1.0
        ))

        Path(f"../../data/metavoice/{wav_name}").write_bytes(wav_file.read_bytes())
        Path(f"../../data/metavoice/{txt_name}").write_text(text)