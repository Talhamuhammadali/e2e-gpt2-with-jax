# Attention & Transformer Block

## Causal Attention Mask
- Lower-triangular matrix of 1s and 0s
- Each token can only attend to itself and tokens before it (never future tokens)
- This makes GPT-2 autoregressive — no cheating by looking ahead
- 0s become `-inf` before softmax, driving those attention weights to zero

## Multi-Head Attention
- Each head learns a different attention pattern independently
  - One head might track syntax, another might track references, etc.
- Each head operates on `n_embd // n_head = 64` dimensions
- All heads run in parallel, outputs are concatenated back to `n_embd`

## Why qkv_features = n_embd
- Not magic — it's a plumbing constraint
- Residual connection (`x = x + attn_output`) requires input and output to be the same size
- Blocks are stacked — block 1 output feeds block 2 input, so dimensions must match

## Transformer Block
- Self-attention sublayer → tokens look at each other
- (MLP/FFN sublayer → processes each token independently — not yet implemented)
- Residual connection adds input back to output at each sublayer

## Full Model Flow
```
token IDs → embeddings (768) → transformer blocks (768→768) → linear (768→50257) → logits
```

## Output Layer
- Linear projection from `n_embd` to `vocab_size`
- Produces one score (logit) per vocabulary token
- Softmax converts logits to probabilities
- Highest probability = predicted next token

## Flax NNX Note
- Must use `nnx.List(...)` not plain Python `list` for module containers
- JAX needs pytree-aware containers to find and track parameters
