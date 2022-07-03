import random

import jax
import jax.numpy as jnp
import numpy as np
import optax
from Crypto.Protocol.SecretSharing import Shamir
from Crypto.Cipher import AES

import utils
import DH


class Client:

    def __init__(self, uid, params, opt, loss, data, epochs=1, t=2, R=2**16 - 1):
        self.id = uid
        self._train_step = train_step(opt, loss)
        self.opt_state = opt.init(params)
        self.data = data
        self.epochs = epochs
        self.params = params
        self.t = t
        self.c = DH.DiffieHellman()
        self.s = DH.DiffieHellman()
        ravelled_params, unraveller = jax.flatten_util.ravel_pytree(params)
        self.unraveller = jax.jit(unraveller)
        self.params_len = len(ravelled_params)
        self.R = R

    def receive_grads(grads):
        self.params = self.unraveller(utils.ravel(self.params) + decode(grads))

    def advertise_keys(self):
        return self.c.gen_public_key(), self.s.gen_public_key()

    def share_keys(self, keylist):
        self.keylist = keylist
        self.b = random.randint(0, self.R)
        s_shares = Shamir.split(self.t, len(keylist), self.s.get_private_key())
        b_shares = Shamir.split(self.t, len(keylist), self.b)
        e = []
        for (v, (cv, _)), ss, bs in zip(keylist.items(), s_shares, b_shares):
            k = self.c.gen_shared_key(cv)
            eu = encrypt_and_digest(bytes(self.id), k)
            ev = encrypt_and_digest(bytes(v), k)
            ess = encrypt_and_digest(ss[1], k)
            ebs = encrypt_and_digest(bs[1], k)
            e.append((eu, ev, ess, ebs))
        return e

    def masked_input_collection(self, e):
        self.e = e
        puvs = []
        for v, (cv, sv) in self.keylist.items():
            if v == self.id:
                puv = jnp.zeros(self.params_len)
            else:
                suv = self.s.gen_shared_key(sv)
                puv = jax.random.randint(jax.random.PRNGKey(int.from_bytes(suv, 'big') % self.R), (self.params_len,), 0, self.R)
                if self.id < v:
                    puv = -puv
            puvs.append(puv)
        pu = jax.random.randint(jax.random.PRNGKey(self.b), (self.params_len,), 0, self.R)
        return encode(self.step()) + pu + sum(puv)

    def step(self):
        params = self.params
        for e in range(self.epochs):
            X, y = next(self.data)
            self.params, self.opt_state, loss = self._train_step(self.params, self.opt_state, X, y)
        return utils.gradient(params, self.params)

    def unmasking(self):
        svu = []
        bvu = []
        for v, evus in enumerate(self.e):
            if self.id != v:
                k = self.c.gen_shared_key(self.keylist[v][0])
                for (ev, eu, ess, ebs) in evus:
                    vprime = decrypt_and_verify(ev, k)
                    uprime = decrypt_and_verify(eu, k)
                    # if u != uprime and v != vprime
                    #    abort
                    # for our case we take it as all clients make it to round 3
                    # svu.append(cipher.decrypt_and_verify(*ss))
                    bvu.append(cipher.decrypt_and_verify(bs, k))
        return svu, bvu


def train_step(opt, loss):

    @jax.jit
    def _apply(params, opt_state, X, y):
        loss_val, grads = jax.value_and_grad(loss)(params, X, y)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    return _apply

def encrypt_and_digest(p, k):
    return AES.new(k, AES.MODE_EAX).encrypt_and_digest(p)


def decrypt_and_verify(ct, k):
    return AES.new(k, AES.MODE_EAX).decrypt_and_verify(*ct)

@jax.jit
def encode(grad):
    return jnp.round(grad * 10_000)


@jax.jit
def decode(grad):
    return grad / 10_000
