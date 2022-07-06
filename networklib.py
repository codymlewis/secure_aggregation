import itertools

import pyseltongue
from Crypto.PublicKey import ECC
from Crypto.Signature import eddsa

import utils
import DH

class Network:
    def __init__(self):
        self.clients = []

    def add_client(self, client):
        self.clients.append(client)

    def setup(self):
        # If interpreted directly, this network would be the TTP
        signing_keys = {}
        verification_keys = {}
        for c in self.clients:
            key = ECC.generate(curve='ed25519')
            signing_keys[c.id] = eddsa.new(key, 'rfc8032')
            verification_keys[c.id] = eddsa.new(key.public_key(), 'rfc8032')
        for c in self.clients:
            c.setup(signing_keys[c.id], verification_keys)

    def begin(self, subject="grads"):
        for c in self.clients:
            self.params_len = c.prepare_for_agg(subject)

    def __call__(self, R):
        keys = self.advertise_keys()
        keylist = {u: (cu, su, sigu) for u, (cu, su, sigu) in keys.items()}
        euvs = self.share_keys(keylist)
        mic = self.masked_input_collection(euvs)
        u3 = set(mic.keys())
        yus = list(mic.values())
        v_sigs = self.consistency_check(u3)
        suvs, buvs = self.unmasking(v_sigs)
        pus, puvs = [], []
        private_keys = []
        for v, suv in enumerate(suvs):
            if suv:
                suv_combined = pyseltongue.points_to_secret_int(suv)
                private_keys.append((v, DH.DiffieHellman(private_key=suv_combined)))
        for (u, pku), (v, (_, pkv, _)) in itertools.product(private_keys, keylist.items()):
            if u != v:
                k = int.from_bytes(pku.gen_shared_key(pkv), 'big') % R
                puvs.append(utils.gen_mask(k, self.params_len, R))
                if u < v:
                    puvs[-1] = -puvs[-1]
        for buv in buvs:
            if buv:
                buv_combined = pyseltongue.points_to_secret_int(buv)
                pus.append(utils.gen_mask(buv_combined, self.params_len, R))
        x = sum(yus) - sum(pus) + sum(puvs)
        return x, len(yus)

    def advertise_keys(self):
        return {c.id: c.advertise_keys() for c in self.clients}

    def share_keys(self, keylist):
        return {c.id: c.share_keys(keylist) for c in self.clients}

    def masked_input_collection(self, euvs):
        return {c.id: c.masked_input_collection(euvs) for c in self.clients}

    def consistency_check(self, u3):
        return {c.id: c.consistency_check(u3) for c in self.clients}

    def unmasking(self, v_sigs):
        svus, bvus = [], []
        for c in self.clients:
            svu, bvu = c.unmasking(v_sigs)
            svus.append(svu)
            bvus.append(bvu)
        buvs = utils.transpose(bvus)
        suvs = utils.transpose(svus)
        return suvs, buvs

    def send_grads(self, grads):
        for c in self.clients:
            c.receive_grads(grads)
