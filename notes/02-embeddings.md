# Embeddings

## Why Embeddings
- Computers can't understand text — need to convert to numbers
- Tokenizer converts text to token IDs (integers)
- Embeddings convert token IDs to dense vectors the model can learn from

## Token Embedding
- Lookup table of shape `(vocab_size, n_embd)` → `(50257, 768)`
- Each token ID maps to a row — a 768-dim vector representing **what** the token is
- "cat" always looks up the same row regardless of position

## Position Embedding
- Lookup table of shape `(block_size, n_embd)` → `(1024, 768)`
- Each position (0, 1, 2, ...) maps to a row — a 768-dim vector representing **where** the token is
- This is GPT-2's hard context limit — no entry for position 1025

## Combining
- Final embedding = token_embedding + position_embedding (element-wise addition)
- Same token at different positions gets different final vectors
- Output shape: `(batch_size, seq_len, n_embd)`
- This combined vector is what enters the first transformer block
