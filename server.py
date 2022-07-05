import einops
import jax
import jax.numpy as jnp
import numpy as np
import optax
from Crypto.Protocol.SecretSharing import Shamir

import utils


class Server:

    def __init__(self, network, params, R=2**16 - 1, rng=np.random.default_rng()):
        self.network = network
        self.grads = jnp.zeros_like(utils.ravel(params))
        self.params_len = len(self.grads)
        self.rng = rng
        self.R = R

    def step(self):
        keys = self.network.advertise_keys()
        keylist = {u: (cu, su) for u, (cu, su) in enumerate(keys)}
        euvs = self.network.share_keys(keylist)
        yus, losses = self.network.masked_input_collection(euvs)
        svus, bvus = self.network.unmasking()
        pu, pvu = [], []
        for svu in svus:
            if svu:
                svu_combined = int.from_bytes(Shamir.combine(svu), 'big') % self.R
                pvu.append(jax.random.randint(jax.random.PRNGKey(svu_combined), (self.params_len,), 0, self.R))
        for bvu in bvus:
            if bvu:
                bvu_combined = int.from_bytes(Shamir.combine(bvu), 'big') % self.R
                pu.append(jax.random.randint(jax.random.PRNGKey(bvu_combined), (self.params_len,), 0, self.R))
        grads = sum(yus) - sum(pu) + sum(pvu)
        self.network.send_grads(grads)
        return np.mean(losses)
