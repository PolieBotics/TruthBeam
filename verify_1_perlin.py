#!/usr/bin/env python3
"""
verify_perlin.py
Check local hash chain + Perlin MSE, storing basic results.

Usage:
  python3 verify_perlin.py [optional /path/to/dataset.h5]
"""

import os
import sys
import glob
import csv
import logging

import numpy as np
import h5py
from PIL import Image

from config import Config
from utils import blake3_hash, create_dir_if_not_exists
from perlin import generate_image_from_hash

logger = logging.getLogger("TruthBeam.VerifyPerlin")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def main():
    if len(sys.argv)>1:
        hdf5_path=sys.argv[1]
        if not os.path.isfile(hdf5_path):
            print(f"{hdf5_path} not found.")
            sys.exit(1)
    else:
        ds_files=glob.glob(os.path.join(Config.NOVEL_DATASETS_DIR,"*","data.h5"))
        if not ds_files:
            print("No data.h5 found in novel_datasets.")
            sys.exit(0)
        print("Available datasets:")
        for i,f in enumerate(ds_files):
            print(f"[{i}] {f}")
        c=input("Select dataset index: ")
        try:
            ci=int(c)
            if ci<0 or ci>=len(ds_files):
                print("Invalid.")
                sys.exit(1)
            hdf5_path=ds_files[ci]
        except:
            print("Invalid input.")
            sys.exit(1)

    logger.info(f"Opening dataset => {hdf5_path}")
    hf=h5py.File(hdf5_path,"r")
    emissions=hf["emissions"]
    recordings=hf["recordings"]
    hashes=hf["hashes"][:]

    init_block=hf.attrs.get("initial_blockhash","")
    final_hash=hf.attrs.get("final_hash","")
    final_tx=hf.attrs.get("final_tx_hash",None)
    logger.info(f"InitialBlock={init_block}, finalHash={final_hash}, finalTx={final_tx}")
    N=emissions.shape[0]
    logger.info(f"Total frames => {N}")

    if init_block:
        init_vec=blake3_hash(bytes.fromhex(init_block))
    else:
        init_vec=b""

    # We'll store results in forensic_outputs
    out_dir=os.path.join(os.path.dirname(hdf5_path),"forensic_outputs")
    create_dir_if_not_exists(out_dir)
    out_csv=os.path.join(out_dir,"stage1_perlin.csv")
    f=open(out_csv,"w",newline="")
    cw=csv.writer(f)
    cw.writerow(["frame_index","hash_ok","perlin_mse"])

    # We'll define a small helper to compute half-res
    def half_resize_emi(arr: np.ndarray)->np.ndarray:
        # arr shape(H,W,3) => Image => half-size => float in [0..1]
        im=Image.fromarray(arr,"RGB")
        hsize=(Config.HALF_EMISSION_HEIGHT, Config.HALF_EMISSION_WIDTH)
        im2=im.resize((hsize[1], hsize[0]))
        arr2=np.array(im2).astype(np.float32)/255.0
        return arr2

    prev=init_vec
    for i in range(N):
        # Hash check
        rec_data=recordings[i]
        comp=blake3_hash(prev, rec_data.tobytes())
        act=hashes[i].tobytes()
        ok=(comp==act)
        if not ok:
            logger.warning(f"Frame {i}: Hash mismatch!")
        prev=act

        # Perlin MSE
        if i==0:
            e_hash=init_vec
        else:
            e_hash=hashes[i-1].tobytes()
        gen_img=generate_image_from_hash(e_hash)  # PIL
        sto_emi=emissions[i]  # (H,W,3) np.uint8
        # half-res
        gen_half=half_resize_emi(np.array(gen_img))
        sto_half=half_resize_emi(sto_emi)
        mse_val=float(((gen_half - sto_half)**2).mean())

        cw.writerow([i,"OK" if ok else "FAIL", f"{mse_val:.4f}"])
        logger.info(f"Frame {i}: hash={('OK' if ok else 'FAIL')}, perlin={mse_val:.4f}")

    f.close()
    hf.close()
    logger.info(f"[verify_perlin] done => {out_csv}")

if __name__=="__main__":
    main()
