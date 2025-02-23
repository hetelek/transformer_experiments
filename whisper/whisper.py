import os
from functools import lru_cache
from typing import Optional, Union
import torchaudio
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def exact_div(x, y):
    assert x % y == 0
    return x // y


# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
# 3000 frames in a mel spectrogram input
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(
    SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(
                array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(os.path.dirname(
        __file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)

    if not torch.is_tensor(audio):
        raise RuntimeError(
            f"Failed to load audio: expected tensor/numpy array, got {type(audio)}")

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH,
                      window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def process_audio_file(file_path: str):
    waveform, sr = torchaudio.load(file_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        raise Exception('invalid sample rate')
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=False)
        raise Exception('invalid sample shape')
    audio = waveform.numpy().reshape(-1)

    for i in range(0, len(audio), N_SAMPLES):
        chunk = audio[i:i + N_SAMPLES]
        chunk = pad_or_trim(chunk, N_SAMPLES)
        chunk = log_mel_spectrogram(chunk)

        # DEBUG: save mel spectrogram to disk
        # plt.figure(figsize=(10, 4))
        # plt.imshow(chunk, aspect='auto', origin='lower', cmap='viridis')
        # plt.colorbar(label='Amplitude (dB)')
        # plt.title('Mel Spectrogram')
        # plt.xlabel('Time Frames')
        # plt.ylabel('Mel Frequency Bins')
        # plt.savefig('mel_spectrogram.png', dpi=300, bbox_inches='tight')
        # plt.close()
        # exit(0)

audio_file = os.path.join(SCRIPT_DIR_PATH, 'samples', 'question-16khz.wav')
process_audio_file(audio_file)
