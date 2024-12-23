import os
import blake3
import torch
import numpy as np

def blake3_hash(*inputs):
    h = blake3.blake3()
    for inp in inputs:
        if isinstance(inp, str):
            inp = inp.encode('utf-8')
        elif isinstance(inp, bytes):
            pass
        else:
            inp = str(inp).encode('utf-8')
        h.update(inp)
    return h.digest()

def create_dir_if_not_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)

def tensor_to_npimg(t):
    # t: C,H,W in [0,1]
    arr = t.detach().cpu().numpy().transpose(1,2,0)*255
    arr = arr.clip(0,255).astype(np.uint8)
    return arr

def derive_address_from_private_key(priv_key_hex):
    # Simple derivation: use web3 if needed
    from web3.auto import w3
    acct = w3.eth.account.from_key(bytes.fromhex(priv_key_hex[2:]))
    return acct.address

def normalize_image_tensor(x):
    return x.clamp(0,1)
