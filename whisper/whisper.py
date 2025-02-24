import os
from functools import lru_cache
from typing import Optional, Union
import torchaudio
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
import time

SCRIPT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def exact_div(x, y):
    assert x % y == 0
    return x // y


# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_MELS = 80
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH) # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token

@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    assert n_mels in {N_MELS, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(os.path.dirname(
        __file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = N_MELS,
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


def forward_pass(audio_chunk: np.ndarray):
    assert audio_chunk.shape[0] == N_SAMPLES

    # convert the audio chunk to a mel spectrogram
    mel_spectrogram = log_mel_spectrogram(audio_chunk)
    print(f'mel_spectrogram: {mel_spectrogram.shape}')
    assert mel_spectrogram.shape == (N_MELS, N_FRAMES) # [80, 3000]
    # TODO: forward pass and decode!
    exit(0)

    # DEBUG: save mel spectrogram to disk
    # plt.figure(figsize=(10, 4))
    # plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    # plt.colorbar(label='Amplitude (dB)')
    # plt.title('Mel Spectrogram')
    # plt.xlabel('Time Frames')
    # plt.ylabel('Mel Frequency Bins')
    # plt.savefig('mel_spectrogram.png', dpi=300, bbox_inches='tight')
    # plt.close()
    # # exit(0)

audio_buffer = None
def process_audio_chunk(audio_chunk: np.ndarray):
    global audio_buffer
    class CircularBuffer:
        def __init__(self, size):
            self.size = size
            self.total_appended = 0
            self.buffer = np.zeros(size, dtype=np.float32)

        def append(self, samples):
            # appends at most `size` samples to the buffer, and returns how many samples were actually appended
            num_samples = len(samples)
            if num_samples >= self.size:
                self.buffer[:] = samples[:self.size] # copy the first `size` samples
                self.total_appended = self.size
                return self.size
            
            if self.total_appended + num_samples >= self.size:
                # shift the existing samples to the left to make space for all the new samples
                self.buffer[:-num_samples] = self.buffer[num_samples:]
                self.buffer[-num_samples:] = samples
                self.total_appended = self.size
            else:
                # append to front
                self.buffer[self.total_appended:self.total_appended + num_samples] = samples
                self.total_appended += num_samples

            return num_samples
        
    if audio_buffer is None:
        audio_buffer = CircularBuffer(N_SAMPLES)

    num_processed = 0
    while num_processed < len(audio_chunk):
        num_processed += audio_buffer.append(audio_chunk[num_processed:])
        forward_pass(audio_buffer.buffer.copy())


def process_audio_file(file_path: str):
    waveform, sr = torchaudio.load(file_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        raise Exception('invalid sample rate')
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=False)
        raise Exception('invalid sample shape')
    audio = waveform.numpy().reshape(-1)
    
    # simulate audio streaming in real-time chunks of 100ms
    streaming_chunk_size = SAMPLE_RATE // 1
    for i in range(0, len(audio), streaming_chunk_size):
        print(f'sending audio chunk: {i}:{i + streaming_chunk_size}')
        chunk = audio[i:i + streaming_chunk_size]
        process_audio_chunk(chunk)
        time.sleep(0.1)

audio_file = os.path.join(SCRIPT_DIR_PATH, 'samples', 'question-16khz.wav')
process_audio_file(audio_file)
