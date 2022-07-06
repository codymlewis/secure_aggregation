from functools import partial
from math import ceil

import jax
import jax.numpy as jnp


@jax.jit
def ravel(params):
    return jax.flatten_util.ravel_pytree(params)[0]


@jax.jit
def gradient(start_params, end_params):
    return ravel(start_params) - ravel(end_params)


@partial(jax.jit, static_argnums=(0, 1, 2))
def gen_mask(key, params_len, R):
    return jax.random.uniform(jax.random.PRNGKey(key), (params_len,), minval=-R, maxval=R)


def transpose(l):
    return [list(i) for i in zip(*l)]

def to_bytes(i):
    return i.to_bytes(ceil(i.bit_length() / 8), 'big')
