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

import tiktoken
import base64

MODEL_NAME = 'tiny' # 'tiny' ('tiny.en' is untested) or 'medium' or 'large-v3-turbo'
LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

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
    def __init__(self, model_name: str, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            dims.n_mels, dims.n_audio_ctx, dims.n_audio_state, dims.n_audio_head, dims.n_audio_layer)
        self.decoder = TextDecoder(
            dims.n_vocab, dims.n_text_ctx, dims.n_text_state, dims.n_text_head, dims.n_text_layer)

        # alignment_heads = {
        #     'tiny.en': [*[0]*6, 1, *[0]*5, 1, *[0]*4, *[1]*6, 0],
        #     'tiny': [*[0]*14, 1, *[0]*3, 1, 0, *[1]*4],
        #     'base.en': [*[0]*27, 1, *[0]*11, 1, 0, 1, *[0]*3, 1, 0, 1],
        #     'base': [*[0]*25, 1, *[0]*8, *[1]*2, *[0]*3, 1, 0, *[1]*2, 0, 1, 0, 1, 0],
        #     'small.en': [*[0]*78, 1, *[0]*5, 1, *[0]*2, 1, *[0]*4, 1, *[0]*5, 1,
        #                  *[0]*2, 1, 0, 1, *[0]*4, 1, *[0]*3, 1, *[0]*3, 1, 0, 1, 0,
        #                  *[1]*4, *[0]*2, 1, *[0]*4, 1, *[0]*2, 1, 0, 1, *[0]*7],
        #     'small': [*[0]*63, 1, *[0]*5, 1, *[0]*26, 1, *[0]*3, 1, *[0]*2, *[1]*2,
        #               *[0]*3, 1, *[0]*6, 1, 0, 1, *[0]*7, 1, *[0]*18],
        #     'medium.en': [*[0]*180, 1, *[0]*44, 1, *[0]*10, 1, 0, 1, *[0]*5, 1, *[0]*11, 1,
        #                   *[0]*3, 1, *[0]*4, 1, *[0]*18, 1, 0, 1, *[0]*8, 1, *[0]*2, 1, *[0]*4,
        #                   1, *[0]*16, 1, *[0]*2, 1, *[0]*5, 1, *[0]*4, 1, *[0]*13, 1, *[0]*35],
        #     'medium': [*[0]*223, 1, *[0]*20, 1, *[0]*10, 1, 0, 1, *[0]*62, 1, *[0]*51, 1, *[0]*11],
        #     'large-v1': [*[0]*199, 1, *[0]*22, 1, 0, 1, *[0]*12, 1, *[0]*209, 1, *[0]*3, 1, *[0]*5,
        #                  1, *[0]*4, 1, *[0]*12, 1, *[0]*164],
        #     'large-v2': [*[0]*212, 1, *[0]*64, 1, *[0]*53, *[1]*3, *[0]*21, *[1]*2, *[0]*7, 1,
        #                  *[0]*6, 1, *[0]*7, 1, *[0]*11, 1, *[0]*30, *[1]*2, *[0]*19, 1, *[0]*5, 1,
        #                  *[0]*2, 1, *[0]*12, 1, 0, 1, *[0]*5, 1, *[0]*31, 1, *[0]*15, 1, *[0]*10, 1,
        #                  *[0]*22, 1, *[0]*84],
        #     'large-v3': [*[0]*140, 1, *[0]*76, 1, *[0]*40, 1, *[0]*13, 1, *[0]*48, 1, *[0]*32, 1,
        #                  *[0]*36, 1, *[0]*32, 1, *[0]*56, 1, *[0]*24, 1, *[0]*133],
        #     'large': [*[0]*140, 1, *[0]*76, 1, *[0]*40, 1, *[0]*13, 1, *[0]*48, 1, *[0]*32, 1,
        #               *[0]*36, 1, *[0]*32, 1, *[0]*56, 1, *[0]*24, 1, *[0]*133],
        #     'large-v3-turbo': [*[0]*44, 1, *[0]*6, 1, *[0]*11, 1, *[0]*2, 1, *[0]*4, 1, *[0]*2, 1, *[0]*5],
        #     'turbo': [*[0]*44, 1, *[0]*6, 1, *[0]*11, 1, *[0]*2, 1, *[0]*4, 1, *[0]*2, 1, *[0]*5],
        # }

    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual())

    def forward(self, mel: Tensor, tokens: Tensor) -> Tensor:
        audio_features = self.encoder(mel)
        return self.decoder(tokens, audio_features)

    @lru_cache(maxsize=None)
    def get_encoding(self):
        tiktoken_vocab_name = 'multilingual' if self.is_multilingual() else 'gpt2'
        vocab_path = os.path.join(
            ASSET_DIR_PATH, f"{tiktoken_vocab_name}.tiktoken")
        ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in (line.split() for line in open(vocab_path) if line)
        }
        n_vocab = len(ranks)
        special_tokens = {}
        num_languages = self.num_languages()
        specials = [
            "<|endoftext|>",
            "<|startoftranscript|>",
            *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
            "<|translate|>",
            "<|transcribe|>",
            "<|startoflm|>",
            "<|startofprev|>",
            "<|nospeech|>",
            "<|notimestamps|>",
            *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
        ]

        for token in specials:
            special_tokens[token] = n_vocab
            n_vocab += 1

        t = tiktoken.Encoding(
            name=os.path.basename(vocab_path),
            explicit_n_vocab=n_vocab,
            pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            mergeable_ranks=ranks,
            special_tokens=special_tokens,
        )
        t.special_tokens = {}
        for special in t.special_tokens_set:
            special_token = t.encode_single_token(special)
            t.special_tokens[special] = special_token
        return t


def load_tiny_model(device):
    # from https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt
    checkpoint = torch.load(os.path.join(ASSET_DIR_PATH, f'{MODEL_NAME}.pt'), map_location=device)
    print(f"dims = {checkpoint['dims']}")
    model = Whisper(MODEL_NAME, ModelDimensions(**checkpoint['dims']))
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model is on: {next(model.parameters()).device}")
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
        device = 'cpu'
        if torch.backends.mps.is_available():
            device = 'mps'
        if torch.cuda.is_available():
            device = 'cuda'
        model = load_tiny_model(device)

    device = next(model.parameters()).device

    # convert the audio chunk to a mel spectrogram
    mel_start = time.perf_counter()
    mel_spectrogram = log_mel_spectrogram(audio_chunk, device=device)
    mel_end = time.perf_counter()
    print(f"mel_time = {(mel_end - mel_start) * 1000:.3f} ms")
    assert mel_spectrogram.shape == (N_MELS, N_FRAMES)  # [80, 3000]

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

    # encode spectrogram
    mel_spectrogram = mel_spectrogram.unsqueeze(0)
    encode_start = time.perf_counter()
    audio_features = model.encoder(mel_spectrogram)
    encode_end = time.perf_counter()
    audio_features.to(device)
    print(f"encode_time = {(encode_end - encode_start) * 1000:.3f} ms")

    # decode in loop until <|endoftext|> token is reached
    sot_sequence = [
        model.get_encoding().special_tokens['<|startoftranscript|>'],
        model.get_encoding().special_tokens['<|en|>'],
        model.get_encoding().special_tokens['<|transcribe|>'],
        model.get_encoding().special_tokens['<|notimestamps|>'],
        # model.get_encoding().special_tokens['<|0.00|>'],
        # *model.get_encoding().encode('I was'),
    ]
    tokens = torch.tensor([sot_sequence], device=device)
    decode_times = []
    while True:
        decode_start = time.perf_counter()
        text_logits = model.decoder(tokens, audio_features, kv_cache=None)
        next_token_id = torch.argmax(text_logits[:, -1], dim=-1).item()
        decode_end = time.perf_counter()
        decode_times.append(decode_end - decode_start)

        # next_token_ids = torch.argmax(text_logits.squeeze(), dim=-1)
        # print(model.get_encoding().decode(next_token_ids.tolist()))
        if next_token_id == model.get_encoding().special_tokens['<|endoftext|>']:
            print()
            break

        print(model.get_encoding().decode([next_token_id]), end='', flush=True)

        if len(decode_times) > 300:
            print('Too many tokens')
            break

        # print(f'decoder_output_shape (text_logits): {text_logits.shape}')
        # print(text_logits[:, -1].shape)
        tokens = torch.cat((tokens, torch.tensor(
            [[next_token_id]], device=device)), dim=1)

    print(
        f"decode_time = {sum(decode_times) * 1000:.3f}ms avg={sum(decode_times) * 1000 / len(decode_times):.3f}ms")

    # exit(0)
    # print()


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
    streaming_chunk_size = exact_div(N_SAMPLES, 1)
    for i in range(0, len(audio), streaming_chunk_size):
        print(f'sending audio chunk: {i}:{i + streaming_chunk_size}')
        chunk = audio[i:i + streaming_chunk_size]
        process_audio_chunk(chunk)
        time.sleep(0.1)


audio_file = os.path.join(SCRIPT_DIR_PATH, 'samples', 'untitled.wav')
process_audio_file(audio_file)
