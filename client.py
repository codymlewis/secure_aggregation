import random

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyseltongue
from Crypto.Cipher import AES

import utils
import DH


class Client:

    def __init__(self, uid, params, opt, loss, data, epochs=1, t=2, R=2**16 - 1):
        self.id = uid
        self._train_step = train_step(opt, loss)
        self._loss = loss
        self.opt_state = opt.init(params)
        self.data = data
        self.epochs = epochs
        self.params = params
        self.t = t
        ravelled_params, unraveller = jax.flatten_util.ravel_pytree(params)
        self.unraveller = jax.jit(unraveller)
        self.R = R

    def prepare_for_agg(self, subject="grads"):
        if subject == "loss":
            self.x = self._loss(self.params, *next(self.data)).reshape(1)
        else:
            self.x = self.step()
        self.params_len = len(self.x)
        return self.params_len

    def step(self):
        params = self.params
        for e in range(self.epochs):
            X, y = next(self.data)
            self.params, self.opt_state = self._train_step(self.params, self.opt_state, X, y)
        return utils.gradient(params, self.params)

    def receive_grads(self, grads):
        self.params = self.unraveller(utils.ravel(self.params) - grads)

    def setup(self, signing_key, verification_keys):
        self.c = DH.DiffieHellman()
        self.s = DH.DiffieHellman()
        self.signing_key = signing_key
        self.verification_keys = verification_keys

    def advertise_keys(self):
        cpk = self.c.gen_public_key()
        spk = self.s.gen_public_key()
        sig = self.signing_key.sign(utils.to_bytes(cpk) + utils.to_bytes(spk))
        return cpk, spk, sig

    def share_keys(self, keylist):
        self.keylist = keylist
        self.u1 = set(keylist.keys())
        assert len(self.u1) >= self.t
        self.b = random.randint(0, self.R)
        s_shares = pyseltongue.secret_int_to_points(self.s.get_private_key(), self.t, len(keylist))
        b_shares = pyseltongue.secret_int_to_points(self.b, self.t, len(keylist))
        e = {}
        for (v, (cv, sv, sigv)), ss, bs in zip(keylist.items(), s_shares, b_shares):
            assert v in self.u1
            ver_msg = utils.to_bytes(cv) + utils.to_bytes(sv)
            self.verification_keys[v].verify(ver_msg, sigv)
            k = self.c.gen_shared_key(cv)
            eu = encrypt_and_digest(self.id.to_bytes(16, 'big'), k)
            ev = encrypt_and_digest(v.to_bytes(16, 'big'), k)
            ess = encrypt_and_digest(utils.to_bytes(ss[1]), k)
            ebs = encrypt_and_digest(utils.to_bytes(bs[1]), k)
            e[v] = (eu, ev, ess, ebs)
        return e

    def masked_input_collection(self, e):
        self.e = e
        self.u2 = set(e.keys())
        assert len(self.u2) >= self.t
        puvs = []
        for v, (cv, sv, _) in self.keylist.items():
            if v == self.id:
                puv = jnp.zeros(self.params_len)
            else:
                suv = int.from_bytes(self.s.gen_shared_key(sv), 'big') % self.R
                puv = utils.gen_mask(suv, self.params_len, self.R)
                if self.id < v:
                    puv = -puv
            puvs.append(puv)
        pu = utils.gen_mask(self.b, self.params_len, self.R)
        return self.x + pu + sum(puvs)

    def consistency_check(self, u3):
        self.u3 = u3
        assert len(self.u3) >= self.t
        return self.signing_key.sign(bytes(u3))

    def unmasking(self, v_sigs):
        for v, sigv in v_sigs.items():
            self.verification_keys[v].verify(bytes(self.u3), sigv)
        svu = []
        bvu = []
        for v, evu in self.e.items():
            ev, eu, ess, ebs = evu[self.id]
            k = self.c.gen_shared_key(self.keylist[v][0])
            uprime = int.from_bytes(decrypt_and_verify(eu, k), 'big')
            vprime = int.from_bytes(decrypt_and_verify(ev, k), 'big')
            assert self.id == uprime and v == vprime
            if v in (self.u2 - self.u3):
                svu.append((self.id + 1, int.from_bytes(decrypt_and_verify(ess, k), 'big')))
            else:
                bvu.append((self.id + 1, int.from_bytes(decrypt_and_verify(ebs, k), 'big')))
        return svu, bvu


def train_step(opt, loss):

    @jax.jit
    def _apply(params, opt_state, X, y):
        grads = jax.grad(loss)(params, X, y)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return _apply

def encrypt_and_digest(p, k):
    return AES.new(k, AES.MODE_EAX, nonce=b'secagg').encrypt_and_digest(p)


def decrypt_and_verify(ct, k):
    return AES.new(k, AES.MODE_EAX, nonce=b'secagg').decrypt_and_verify(*ct)
