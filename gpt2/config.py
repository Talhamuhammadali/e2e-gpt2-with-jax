"""GPT-2 model configurations for different sizes."""
import tiktoken
from dataclasses import dataclass

gpt_tokenizer = tiktoken.get_encoding("gpt2")
GPT2_VOCAB_SIZE = gpt_tokenizer.n_vocab

@dataclass
class GPT2Config:
    """Hyperparameters that define a GPT-2 model variant."""

    vocab_size: int = GPT2_VOCAB_SIZE   # GPT-2 BPE vocabulary size
    block_size: int = 1024    # maximum context / sequence length
    n_layer: int = 12         # number of transformer blocks
    n_head: int = 12          # number of attention heads
    n_embd: int = 768         # embedding / hidden dimension


# ── Named presets (matching the original OpenAI GPT-2 releases) ───────────────

GPT2_SMALL = GPT2Config(n_layer=12, n_head=12, n_embd=768)
GPT2_MEDIUM = GPT2Config(n_layer=24, n_head=16, n_embd=1024)
GPT2_LARGE = GPT2Config(n_layer=36, n_head=20, n_embd=1280)
GPT2_XL = GPT2Config(n_layer=48, n_head=25, n_embd=1600)
