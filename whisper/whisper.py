import os
from functools import lru_cache
from typing import Optional, Union
import torchaudio
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
import time
from torch import Tensor

SCRIPT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
ASSET_DIR_PATH = os.path.join(SCRIPT_DIR_PATH, 'assets')


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
# 3000 frames in a mel spectrogram input
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(
    SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


# Model definition
class ModelDimensions:
    def __init__(self, n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer,
                 n_vocab, n_text_ctx, n_text_state, n_text_head, n_text_layer):
        self.n_mels = n_mels
        self.n_audio_ctx = n_audio_ctx
        self.n_audio_state = n_audio_state
        self.n_audio_head = n_audio_head
        self.n_audio_layer = n_audio_layer
        self.n_vocab = n_vocab
        self.n_text_ctx = n_text_ctx
        self.n_text_state = n_text_state
        self.n_text_head = n_text_head
        self.n_text_layer = n_text_layer


class LayerNorm(torch.nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(torch.nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.linear(x, self.weight.to(x.dtype),
                                          None if self.bias is None else self.bias.to(x.dtype))


class Conv1d(torch.nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))


def sinusoids(length, channels, max_timescale=10000):
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment *
                               torch.arange(channels // 2))
    scaled_time = torch.arange(
        length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None):
        q = self.query(x)
        k = self.key(x if xa is None else xa)
        v = self.value(x if xa is None else xa)
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        qk = (q * scale) @ (k * scale).transpose(-1, -2)
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        w = torch.nn.functional.softmax(qk, dim=-1).to(q.dtype)
        out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.out(out), None


class ResidualAttentionBlock(torch.nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)
        self.cross_attn = MultiHeadAttention(
            n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None
        n_mlp = n_state * 4
        self.mlp = torch.nn.Sequential(
            Linear(n_state, n_mlp), torch.nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x),
                                    xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(torch.nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3,
                            stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        self.blocks = torch.nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        x = torch.nn.functional.gelu(self.conv1(x))
        x = torch.nn.functional.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        assert x.shape[1] == 1500, f"Expected time dim 1500, got {x.shape[1]}"
        x = (x + self.positional_embedding).to(x.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        return x


class TextDecoder(torch.nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(n_vocab, n_state)
        self.positional_embedding = torch.nn.Parameter(
            torch.empty(n_ctx, n_state))
        self.blocks = torch.nn.ModuleList([ResidualAttentionBlock(
            n_state, n_head, cross_attention=True) for _ in range(n_layer)])
        self.ln = LayerNorm(n_state)
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        offset = 0
        x = self.token_embedding(
            x) + self.positional_embedding[offset:offset + x.shape[-1]]
        x = x.to(xa.dtype)
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()
        return logits


class Whisper(torch.nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            dims.n_mels, dims.n_audio_ctx, dims.n_audio_state, dims.n_audio_head, dims.n_audio_layer)
        self.decoder = TextDecoder(
            dims.n_vocab, dims.n_text_ctx, dims.n_text_state, dims.n_text_head, dims.n_text_layer)

    def forward(self, mel: Tensor, tokens: Tensor) -> Tensor:
        audio_features = self.encoder(mel)
        return self.decoder(tokens, audio_features)


def load_tiny_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    dims = ModelDimensions(
        n_mels=80, n_audio_ctx=1500, n_audio_state=384, n_audio_head=6, n_audio_layer=4,
        n_vocab=51865, n_text_ctx=448, n_text_state=384, n_text_head=6, n_text_layer=4
    )
    model = Whisper(dims)
    # from https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt
    checkpoint = torch.load(os.path.join(
        ASSET_DIR_PATH, "tiny.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    return model


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    assert n_mels in {N_MELS, 128}, f"Unsupported n_mels: {n_mels}"
    with np.load(os.path.join(ASSET_DIR_PATH, "mel_filters.npz"), allow_pickle=False) as f:
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


global model


def forward_pass(audio_chunk: np.ndarray):
    global model

    assert audio_chunk.shape[0] == N_SAMPLES
    model = None
    if model is None:
        model = load_tiny_model()

    # convert the audio chunk to a mel spectrogram
    mel_spectrogram = log_mel_spectrogram(audio_chunk)
    print(f'mel_spectrogram: {mel_spectrogram.shape}')
    assert mel_spectrogram.shape == (N_MELS, N_FRAMES)  # [80, 3000]
    mel_spectrogram = mel_spectrogram.unsqueeze(0)
    print(model.encoder(mel_spectrogram).shape)
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
                # copy the first `size` samples
                self.buffer[:] = samples[:self.size]
                self.total_appended = self.size
                return self.size

            if self.total_appended + num_samples >= self.size:
                # shift the existing samples to the left to make space for all the new samples
                self.buffer[:-num_samples] = self.buffer[num_samples:]
                self.buffer[-num_samples:] = samples
                self.total_appended = self.size
            else:
                # append to front
                self.buffer[self.total_appended:self.total_appended +
                            num_samples] = samples
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
