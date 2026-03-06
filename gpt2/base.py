"""Main GPT-2 model definition and utilities."""
import jax
import jax.numpy as jnp
import flax.nnx as nnx


def causal_attention_mask(seq_len: int) -> jnp.ndarray:
    """Create a causal attention mask for a sequence of the given length."""
    return jnp.tril(jnp.ones((seq_len, seq_len)))

class TokenAndPositionEmbedding(nnx.Module):
    """Combines token and position embeddings into a single representation."""

    def __init__(self, block_size: int, vocab_size: int, n_embd: int, *, rngs: nnx.rnglib.Rngs):
        """
        Args:
            block_size: max sequence length (1024 in GPT-2), sets the positional embedding table size
            vocab_size: BPE vocabulary size (50257 in GPT-2), sets the token embedding table size
            n_embd: hidden dimension (768 in GPT-2 Small), the vector size each token is represented as
        """
        self.token_emb = nnx.Embed(vocab_size, n_embd, rngs=rngs)
        self.pos_emb = nnx.Embed(block_size, n_embd, rngs=rngs)
    
    def __call__(self, x):
        if x.shape[-1] > self.pos_emb.num_embeddings:
            raise ValueError(f"Input sequence length {x.shape[-1]} exceeds maximum {self.pos_emb.num_embeddings}")
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        return self.token_emb(x) + self.pos_emb(positions)
    

class TransformerBlock(nnx.Module):
    """A single transformer block consisting of self-attention and feed-forward layers."""

    def __init__(self, n_embd: int, n_head: int, *, rngs: nnx.rnglib.Rngs):
        """
        Args:
            n_embd: hidden dimension, used as input/output size for attention
            n_head: number of parallel attention heads, each gets n_embd // n_head dimensions
        """
        self.attn = nnx.MultiHeadAttention(
            num_heads=n_head,
            in_features=n_embd,
            qkv_features=n_embd,
            out_features=n_embd,
            decode=False,
            rngs=rngs,
        )
    
    def __call__(self, x, mask=None):
        attn_output = self.attn(x, mask=mask)
        x = x + attn_output
        return x
    
    
class GPT2Model(nnx.Module):
    """The main GPT-2 model class, consisting of multiple transformer blocks."""

    def __init__(
        self,
        block_size: int,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        *,
        rngs: nnx.rnglib.Rngs
    ):
        """
        Args:
            block_size: max context length (1024), tokens beyond this are not representable
            vocab_size: size of the BPE token dictionary (50257)
            n_embd: hidden dimension flowing through the entire model (768/1024/1280/1600)
            n_head: attention heads per transformer block (12/16/20/25)
            n_layer: number of stacked transformer blocks — the depth of the model (12/24/36/48)
        """
        self.block_size = block_size
        self.embedding = TokenAndPositionEmbedding(
            block_size=block_size,
            vocab_size=vocab_size,
            n_embd=n_embd,
            rngs=rngs,
        )
        self.transformer_blocks = nnx.List([
            TransformerBlock(
                n_embd=n_embd,
                n_head=n_head,
                rngs=rngs,
            ) for _ in range(n_layer)
        ])
        self.output_layer = nnx.Linear(n_embd, vocab_size, use_bias=False, rngs=rngs)
    
    def __call__(self, x):
        mask = causal_attention_mask(x.shape[1])
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x, mask=mask)
        logits = self.output_layer(x)
        return logits