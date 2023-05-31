import jax
import chex
import math
import optax
import equinox as eqx
import jax.random as jr
import jax.numpy as jnp
from jax import jit
from jax import vmap
from typing import Optional
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import PyTree
from jaxtyping import Integer


class Head(eqx.Module):
    key_net: eqx.nn.Linear
    query_net: eqx.nn.Linear
    value_net: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    qk_size: int 

    def __init__(
        self,
        query_size: int,
        key_size: Optional[int] = None,
        value_size: Optional[int] = None,
        qk_size: Optional[int] = None,
        vo_size: Optional[int] = None,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        dropout_p: float = 0.2,
        *,
        key: jax.random.PRNGKey
    ):
        if key_size is None:
            key_size = query_size
        if value_size is None:
            value_size = query_size
        if qk_size is None:
            qk_size = query_size
        if vo_size is None:
            vo_size = query_size

        key_key_net, key_query_net, key_value_net = jr.split(key, num=3)
        self.qk_size = qk_size
        self.key_net = eqx.nn.Linear(
            in_features=key_size,
            out_features=qk_size,
            use_bias=use_key_bias,
            key=key_key_net,
        )
        self.query_net = eqx.nn.Linear(
            in_features=query_size,
            out_features=qk_size,
            use_bias=use_query_bias,
            key=key_query_net,
        )
        self.value_net = eqx.nn.Linear(
            in_features=value_size,
            out_features=vo_size,
            use_bias=use_value_bias,
            key=key_value_net,
        )
        self.dropout = eqx.nn.Dropout(p=dropout_p)

    def __call__(self, x: Float[Array, "T C"], *, key: jr.PRNGKey) -> Float[Array, "T H"]:
        chex.assert_rank(x, 2)

        T = x.shape[0]
        k: Float[Array, "T H"] = vmap(self.key_net)(x)
        q: Float[Array, "T H"] = vmap(self.query_net)(x)
        v: Float[Array, "T H"] = vmap(self.value_net)(x)
        tril: Float[Array, "T T"] = jnp.tril(jnp.ones((T, T)))
        wei: Float[Array, "T T"] = (k @ q.transpose(-1, -2)) * (self.qk_size**-0.5)
        wei = jnp.where(tril == 0, float("-inf"), wei)
        wei = jax.nn.softmax(wei, axis=-1)
        wei = self.dropout(wei, key=key)
        out: Float[Array, "T H"] = wei @ v
        return out
      
      
class MultiHead(eqx.Module):
    head_list: typing.List[HeadLayer]
    proj_layer: eqx.nn.Linear
    dropout_layer: eqx.nn.Linear

    def __init__(
        self,
        num_heads: int,
        query_size: int,
        key_size: Optional[int] = None,
        value_size: Optional[int] = None,
        output_size: Optional[int] = None,
        qk_size: Optional[int] = None,
        vo_size: Optional[int] = None,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
        dropout_p: float = 0.2,
        *,
        key: jax.random.PRNGKey
    ) -> None:

        if key_size is None:
            key_size = query_size
        if value_size is None:
            value_size = query_size
        if qk_size is None:
            qk_size = query_size
        if vo_size is None:
            vo_size = query_size        
        if output_size is None:
            output_size = query_size
                
        *keys, key_proj = jr.split(key, num_heads + 1)
        self.head_list = [
            Head(
                query_size=query_size, 
                key_size=key_size, 
                value_size=value_size, 
                qk_size=qk_size//num_heads, 
                vo_size=vo_size//num_heads,
                use_key_bias=use_key_bias,
                use_query_bias=use_query_bias,
                use_value_bias=use_value_bias,
                dropout_p=dropout_p,
                key=keys[e]
        )
            for e in range(num_heads)
        ]
        self.proj_layer = eqx.nn.Linear(
            vo_size, 
            output_size, 
            use_bias=use_output_bias, key=key_proj
        )

        self.dropout_layer = eqx.nn.Dropout(0.2)

    def __call__(
        self, x: Float[Array, "T n_embd"], *, key
    ) -> Float[Array, "T C"]:
        keys = jr.split(key, num = len(self.head_list) + 1)
        y = jnp.concatenate([nn(x, key=keys[e]) for e, nn in enumerate(self.head_list)], axis=-1)
        y = vmap(self.proj_layer)(y)
        y = self.dropout_layer(y, key=keys[-1])
        return y
      
      
class FeedForward(eqx.Module):
    model: PyTree

    def __init__(
            self, 
            input_size, 
            intra_size: typing.Optional[int] = None, 
            dropout_size: float = 0.2,
            *, 
            key
        ):
        if intra_size is None:
            intra_size = input_size * 4
    
        keys = jr.split(key, num=2)
        self.model = eqx.nn.Sequential(
            [
                eqx.nn.Linear(input_size, intra_size, key=keys[0]),
                eqx.nn.Lambda(jax.nn.gelu),
                eqx.nn.Linear(intra_size, input_size, key=keys[1]),
                eqx.nn.Dropout(p=dropout_size)
            ]
        )

    def __call__(self, x, key):
        return self.model(x, key=key)
    
    

class Block(eqx.Module):
    msa: PyTree
    ffw: PyTree
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm

    def __init__(
            self,    
            num_heads: int,
            query_size: int, 
            *, key
        ):
        key_msa, key_ffw = jr.split(key, 2)
        self.msa = MultiHead(num_heads=num_heads, query_size=query_size, key=key_msa)
        self.ffw = FeedForward(query_size,  key=key_ffw)
        self.ln1 = eqx.nn.LayerNorm(query_size)
        self.ln2 = eqx.nn.LayerNorm(query_size)

    def __call__(self, x, key: jr.PRNGKey):
        T = x.shape[0]
        keys = jr.split(key, T)
        x = x + self.msa(vmap(self.ln1)(x), key=key)
        x = x + vmap(self.ffw, in_axes=(0, 0))(vmap(self.ln2)(x), keys)
        return x
      
           
 class MultiBlock(eqx.Module):
    model: eqx.nn.Sequential
    ln: eqx.nn.LayerNorm
    
    def __init__(
        self, 
        num_blocks: int,
        num_heads: int,
        query_size: int, 
        *, 
        key
    ):
        keys = jr.split(key, num_blocks)
        self.model = eqx.nn.Sequential([
            Block(num_heads, query_size, key=keys[i]) for i in range(num_blocks)
        ])
        self.ln = eqx.nn.LayerNorm(query_size)

    def __call__(self, x, *, key):
        y = self.model(x, key=key)
        return vmap(self.ln)(y)

    
  
class LLM(eqx.Module):
    multiblock: MultiBlock
    token_embedding: eqx.nn.Embedding
    position_embedding: eqx.nn.Embedding
    mlp: eqx.nn.MLP
    ctx_len: int

    def __init__(
        self,
        vocab_size: int,        # Size of the Vocabulary.
        embd: int,              # Embedding Dimension.
        ctx_len: int,           # Context Length (T), max number of tokens
        num_blocks: int = 4,
        num_heads: int = 4,
        *,
        key: jax.random.PRNGKey
    ):
        key_token_embedding, key_position_embedding, key_mlp, key_block = jr.split(key, num=4)

        self.ctx_len = ctx_len
        self.token_embedding = eqx.nn.Embedding(
            num_embeddings=vocab_size, embedding_size=embd, key=key_token_embedding
        )

        self.position_embedding = eqx.nn.Embedding(
            num_embeddings=ctx_len, embedding_size=embd, key=key_position_embedding
        )

        self.multiblock = MultiBlock(num_blocks=num_blocks, num_heads=num_heads, query_size=embd, key=key_block)

        self.mlp = eqx.nn.MLP(
            embd,
            vocab_size,
            width_size=embd * 4,
            depth=2,
            activation=jax.nn.gelu,
            final_activation=lambda x: x,  # Identity
            key=key_mlp,
        )

    def __call__(self, x: Float[Array, "T"], key) -> Float[Array, "T vocab_size"]:
        T = x.shape[0]
        pos_token = jnp.arange(T)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos_token)
        emb = tok_emb + pos_emb
        emb = self.multiblock(emb, key=key)
        out = vmap(self.mlp)(emb)
        return out
