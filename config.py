"""GPT-2 model configurations for different sizes."""
from dataclasses import dataclass


@dataclass
class GPT2Config:
    """Hyperparameters that define a GPT-2 model variant."""

    # Model architecture
    vocab_size: int = 50257   # GPT-2 BPE vocabulary size (unchanged across sizes)
    block_size: int = 1024    # maximum context / sequence length
    n_layer: int = 12         # number of transformer blocks
    n_head: int = 12          # number of attention heads
    n_embd: int = 768         # embedding / hidden dimension
    dropout: float = 0.1      # dropout probability (0.0 = disabled)

    # Training defaults
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 2000
    max_iters: int = 600_000

    # Checkpointing / logging
    out_dir: str = "checkpoints"
    eval_interval: int = 2000
    eval_iters: int = 200
    log_interval: int = 1
    save_interval: int = 5000
    seed: int = 42


# ── Named presets (matching the original OpenAI GPT-2 releases) ───────────────

# GPT-2 Small  – 117 M parameters
GPT2_SMALL = GPT2Config(
    n_layer=12,
    n_head=12,
    n_embd=768,
)

# GPT-2 Medium – 345 M parameters
GPT2_MEDIUM = GPT2Config(
    n_layer=24,
    n_head=16,
    n_embd=1024,
)

# GPT-2 Large  – 762 M parameters
GPT2_LARGE = GPT2Config(
    n_layer=36,
    n_head=20,
    n_embd=1280,
)

# GPT-2 XL     – 1.5 B parameters
GPT2_XL = GPT2Config(
    n_layer=48,
    n_head=25,
    n_embd=1600,
)
