#!/usr/bin/env python3
"""
verify_recordings.py - Example verification script producing a 64x64 latent authenticity map.
(1) We assume a "LatentDiscriminator" with no stride=2 (all stride=1).
(2) We forcibly upsample the AE's fused latent to 64x64 before passing to disc.
(3) The final disc_map is 64x64, which we colorize for a full-resolution heatmap.
"""

import os
import sys
import csv
import glob
import logging

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from config import Config
from models import AdversarialAutoencoder, LatentDiscriminator, LatentGenerator
from utils import create_dir_if_not_exists
import torchvision.models as models

logger = logging.getLogger("TruthBeam.VerifyRecordings")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

########################################
# Colormap settings
########################################
COLORMAP_MODE    = "jet"  # or "heat"
COLORMAP_DEGREE  = 1.0    # exponent/gamma

def color_map(val: float)->(int,int,int):
    """A simple piecewise colormap from val in [0..1]."""
    val = max(0.0, min(1.0, val))
    val = val ** COLORMAP_DEGREE
    if COLORMAP_MODE=="jet":
        # simplified "jet"
        if val<0.2:
            alpha=(val-0.0)/0.2
            r=0
            g=int(255*alpha)
            b=255
        elif val<0.4:
            alpha=(val-0.2)/0.2
            r=0
            g=255
            b=int(255*(1-alpha))
        elif val<0.6:
            alpha=(val-0.4)/0.2
            r=int(255*alpha)
            g=255
            b=0
        elif val<0.8:
            alpha=(val-0.6)/0.2
            r=255
            g=int(255*(1-alpha))
            b=0
        else:
            alpha=(val-0.8)/0.2
            r=255
            g=0
            b=int(255*alpha)
        return (r,g,b)
    else:
        # "heat": black->red->yellow->white
        alpha=val
        if alpha<0.5:
            sc=alpha*2
            r=int(255*sc)
            g=0
            b=0
        else:
            sc2=(alpha-0.5)*2
            if sc2<0.5:
                sc3=sc2*2
                r=255
                g=int(255*sc3)
                b=0
            else:
                sc3=(sc2-0.5)*2
                r=255
                g=255
                b=int(255*sc3)
        return (r,g,b)

def apply_color_map(arr_norm: np.ndarray)->np.ndarray:
    """arr_norm shape(H,W), produce (H,W,3)."""
    H,W=arr_norm.shape
    out=np.zeros((H,W,3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            out[y,x]=color_map(arr_norm[y,x])
    return out

def pad_and_center(emi, rec):
    B,Ce,He,We=emi.shape
    B,Cr,Hr,Wr=rec.shape
    Ht=max(He,Hr)
    Wt=max(We,Wr)
    # emi
    dH_e=Ht-He
    top_e=dH_e//2
    bot_e=dH_e-top_e
    dW_e=Wt-We
    left_e=dW_e//2
    rig_e=dW_e-left_e
    emi_p=F.pad(emi,(left_e,rig_e,top_e,bot_e))
    # rec
    dH_r=Ht-Hr
    top_r=dH_r//2
    bot_r=dH_r-top_r
    dW_r=Wt-Wr
    left_r=dW_r//2
    rig_r=dW_r-left_r
    rec_p=F.pad(rec,(left_r,rig_r,top_r,bot_r))
    return emi_p, rec_p

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg=models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:36].eval()
        for p in vgg.parameters():
            p.requires_grad=False
        self.vgg=vgg
        self.mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
        self.std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    def forward(self,x):
        dev=x.device
        x=(x-self.mean.to(dev))/self.std.to(dev)
        return self.vgg(x)

def vgg_loss(feat_ex, x, y):
    fx=feat_ex(x)
    fy=feat_ex(y)
    return F.l1_loss(fx,fy)

def main():
    # must have stage1_perlin.csv
    if len(sys.argv)>1:
        hdf5_path=sys.argv[1]
        if not os.path.isfile(hdf5_path):
            print(f"{hdf5_path} not found.")
            sys.exit(1)
    else:
        ds_files=glob.glob(os.path.join(Config.NOVEL_DATASETS_DIR,"*","data.h5"))
        if not ds_files:
            print("No data in novel_datasets.")
            sys.exit(0)
        print("Available datasets:")
        for i,f in enumerate(ds_files):
            print(f"[{i}] {f}")
        c=input("Select dataset index: ")
        ci=int(c)
        hdf5_path=ds_files[ci]

    # read stage1
    out_dir=os.path.join(os.path.dirname(hdf5_path),"forensic_outputs")
    stg1_csv=os.path.join(out_dir,"stage1_perlin.csv")
    if not os.path.isfile(stg1_csv):
        print("Need stage1_perlin.csv => run verify_perlin.py")
        sys.exit(1)

    # load that
    logger.info(f"Reading stage1 => {stg1_csv}")
    frame_info={}
    with open(stg1_csv,"r") as f:
        rr=csv.reader(f)
        head=next(rr)
        for row in rr:
            i_s, ok_s, pm_s = row
            idx=int(i_s)
            frame_info[idx]=(ok_s, float(pm_s))

    # open hdf5
    hf=h5py.File(hdf5_path,"r")
    emissions=hf["emissions"]
    recordings=hf["recordings"]
    N=emissions.shape[0]

    final_csv=os.path.join(out_dir,"stage2_AE_latent.csv")
    ff=open(final_csv,"w",newline="")
    cw=csv.writer(ff)
    cw.writerow([
      "frame","hash_ok","perlin_mse",
      "L1_em","L1_rec","diff_mean",
      "VGG_em","VGG_rec",
      "latent_auth","latent_gen_auth","latent_VGG_em","latent_VGG_rec"
    ])

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load AE & latent
    ae=None
    disc_lat=None
    gen_lat=None
    if os.path.isfile("aae_model_latest.pth"):
        logger.info("Load AE => aae_model_latest.pth")
        ae=AdversarialAutoencoder().eval().to(device)
        ae.load_state_dict(torch.load("aae_model_latest.pth", map_location=device))
    else:
        logger.warning("No aae_model_latest => skip AE checks")

    if os.path.isfile("latent_gan_generator_latest.pth") and os.path.isfile("latent_gan_discriminator_latest.pth"):
        logger.info("Load latent => 'latent_gan_*_latest.pth'")
        disc_lat=LatentDiscriminator().eval().to(device)
        gen_lat=LatentGenerator(noise_dim=Config.NOISE_DIM).eval().to(device)
        disc_lat.load_state_dict(torch.load("latent_gan_discriminator_latest.pth",map_location=device))
        gen_lat.load_state_dict(torch.load("latent_gan_generator_latest.pth",map_location=device))

    feat_ex=None
    if ae:
        feat_ex=VGGFeatureExtractor().to(device)

    # subdir for heatmaps
    hm_dir=os.path.join(out_dir,"heatmaps_stage2")
    create_dir_if_not_exists(hm_dir)

    # define half-res
    def half_emi(arr):
        # arr => shape(H,W,3) => PIL => half => torch
        im=Image.fromarray(arr,"RGB")
        tsize=(Config.HALF_EMISSION_HEIGHT, Config.HALF_EMISSION_WIDTH)
        im2=im.resize((tsize[1],tsize[0]))
        arr2=np.array(im2).astype(np.float32)/255.0
        t=torch.from_numpy(arr2).permute(2,0,1).unsqueeze(0)
        return t
    def half_rec(arr):
        # rec => BGR => convert => half
        arr_rgb=arr[...,::-1]
        im=Image.fromarray(arr_rgb,"RGB")
        tsize=(Config.HALF_RECORDING_HEIGHT, Config.HALF_RECORDING_WIDTH)
        im2=im.resize((tsize[1],tsize[0]))
        arr2=np.array(im2).astype(np.float32)/255.0
        t=torch.from_numpy(arr2).permute(2,0,1).unsqueeze(0)
        return t

    for i in range(N):
        ok_s, pm_val=frame_info[i]
        L1e="N/A"; L1r="N/A"; df_m="N/A"
        ve="N/A"; vr="N/A"
        la="N/A"; lg="N/A"; lve="N/A"; lvr="N/A"

        e_np=emissions[i]
        r_np=recordings[i]
        if ae:
            e_t=half_emi(e_np).to(device)
            r_t=half_rec(r_np).to(device)
            e_p, r_p=pad_and_center(e_t,r_t)
            with torch.no_grad():
                re, rr=ae(e_p, r_p)
            re_c=re[:,:,: e_t.shape[2], : e_t.shape[3]]
            rr_c=rr[:,:,: r_t.shape[2], : r_t.shape[3]]
            L1e=float(F.l1_loss(re_c,e_t).item())
            L1r=float(F.l1_loss(rr_c,r_t).item())

            diff=(re_c - e_t).abs().mean(1,keepdim=True)
            df_m=float(diff.mean().item())

            # color-coded diff
            dn=diff[0,0].cpu().numpy()
            mn=dn.min(); mx=dn.max()
            if mx>mn+1e-8:
                norm=(dn-mn)/(mx-mn)
            else:
                norm=dn*0
            diff_rgb=apply_color_map(norm)
            diff_path=os.path.join(hm_dir,f"frame_{i}_diffmap.png")
            Image.fromarray(diff_rgb,"RGB").save(diff_path)

            if feat_ex:
                # vgg
                TS=(256,256)
                re_v=F.interpolate(re_c, TS, mode='bilinear')
                rr_v=F.interpolate(rr_c, TS, mode='bilinear')
                e_v =F.interpolate(e_t,  TS, mode='bilinear')
                r_v =F.interpolate(r_t,  TS, mode='bilinear')
                ve=float(F.l1_loss(feat_ex(e_v), feat_ex(re_v)).item())
                vr=float(F.l1_loss(feat_ex(r_v), feat_ex(rr_v)).item())

            if disc_lat and gen_lat:
                with torch.no_grad():
                    ze=ae.emission_encoder(e_p)
                    zr=ae.recording_encoder(r_p)
                    fused=ae.fusion(torch.cat([ze,zr],1))
                    fused=ae.post_fusion(fused)
                    # force 64x64 upsample
                    fused_64=F.interpolate(fused,(64,64),mode='bilinear',align_corners=False)
                    disc_map=disc_lat(fused_64)
                    la=float(torch.sigmoid(disc_map).mean().item())

                    # color disc_map => shape(1,1,64,64)
                    disc_np=disc_map[0,0].cpu().numpy()
                    dmin=disc_np.min(); dmax=disc_np.max()
                    if dmax> dmin+1e-8:
                        disc_norm=(disc_np-dmin)/(dmax-dmin)
                    else:
                        disc_norm=disc_np*0
                    disc_col=apply_color_map(disc_norm)
                    disc_path=os.path.join(hm_dir,f"frame_{i}_latentmap.png")
                    Image.fromarray(disc_col,"RGB").save(disc_path)

                    # gen
                    noise=torch.randn(1,Config.NOISE_DIM,device=device)
                    fak=gen_lat(noise)
                    # disc map => shape(1,1,?,?)
                    d2=disc_lat(fak)
                    lg=float(torch.sigmoid(d2).mean().item())

                    # decode
                    efk=ae.emission_decoder(fak)
                    rfk=ae.recording_decoder(fak)
                    if feat_ex:
                        efv=F.interpolate(efk,(256,256),mode='bilinear')
                        rfv=F.interpolate(rfk,(256,256),mode='bilinear')
                        ev=F.interpolate(e_t,(256,256),mode='bilinear')
                        rv=F.interpolate(r_t,(256,256),mode='bilinear')
                        lve=float(F.l1_loss(feat_ex(ev), feat_ex(efv)).item())
                        lvr=float(F.l1_loss(feat_ex(rv), feat_ex(rfv)).item())

        cw.writerow([
            i, ok_s, f"{pm_val:.4f}",
            L1e, L1r, df_m,
            ve, vr,
            la, lg, lve, lvr
        ])
        logger.info(f"Frame {i}: hash={ok_s}, pm={pm_val:.4f}, AE(L1={L1e:.4f}/{L1r:.4f}, diff={df_m:.4f}), latent={la:.4f}/{lg:.4f}")

    ff.close()
    hf.close()
    logger.info(f"[verify_recordings] done => {final_csv}")

if __name__=="__main__":
    main()
