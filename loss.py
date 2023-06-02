import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
from jax import jit
from jax import vmap
from jaxtyping import Array
from jaxtyping import PyTree


@jit
def cross_entropy_with_logits(logits, labels):
    log_probs = logits - logsumexp(logits, axis=-1, keepdims=True)
    log_probs_of_class = jnp.take_along_axis(log_probs, labels[:, None], axis=-1)
    return -jnp.mean(log_probs_of_class)

  
@eqx.filter_jit
def loss(model, x, y, key):
    def batched_loss(x, y):
        l = vmap(cross_entropy_with_logits, in_axes=(0, 0))
        return l(x, y).mean()
    keyb = jr.split(key, num=x.shape[0])
    logits = vmap(model)(x, key=keyb)
    losses = batched_loss(logits, y)
    return losses
  

def create_step(optim):
    @eqx.filter_jit
    def make_step(
        model: PyTree,
        opt_state: PyTree,
        x: Array,
        y: Array,
        *,
        key
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y, key=key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, {"loss": loss_value}
    return make_step
