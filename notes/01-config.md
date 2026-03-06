# GPT-2 Config

## Architecture Parameters
- `vocab_size (50257)` — BPE tokenizer dictionary size, same across all GPT-2 sizes
- `block_size (1024)` — max context window, hard limit due to learned positional embeddings
- `n_layer` — number of stacked transformer blocks (depth), 12/24/36/48
- `n_head` — parallel attention heads per block, 12/16/20/25
- `n_embd` — hidden dimension (width), 768/1024/1280/1600

## Key Relationships
- `n_embd` must be divisible by `n_head` — each head gets `n_embd // n_head` dims (always 64 in GPT-2)
- Parameter count scales roughly as `12 * n_layer * n_embd^2`
- Bigger model = both deeper (more layers) and wider (larger n_embd)

## Positional Encoding
- GPT-2 uses absolute learned positional embeddings (fixed table of 1024 vectors)
- Modern LLMs mostly use RoPE (rotary) — no fixed table, can extend context length
- ALiBi is another alternative (used in BLOOM, MPT)
