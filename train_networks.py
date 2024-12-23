import os
import glob
import h5py
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from config import Config
from models import AdversarialAutoencoder, Discriminator, LatentDiscriminator, LatentGenerator
from utils import create_dir_if_not_exists, tensor_to_npimg, derive_address_from_private_key
import logging
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger("TruthBeam.Train")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class MultiHDF5Dataset(Dataset):
    def __init__(self, files):
        self.files=files
        self.index_map=[]
        self.datasets=[]
        for f in self.files:
            h=h5py.File(f,"r")
            n=h["emissions"].shape[0]
            self.datasets.append((f,h,n))
            for i in range(n):
                self.index_map.append((len(self.datasets)-1,i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self,idx):
        ds_i,f_i=self.index_map[idx]
        f,h,n=self.datasets[ds_i]
        emi=h["emissions"][f_i]
        rec=h["recordings"][f_i]

        emi_img=Image.fromarray(emi,"RGB").resize((Config.HALF_EMISSION_WIDTH, Config.HALF_EMISSION_HEIGHT), Image.LANCZOS)
        rec_rgb=rec[..., ::-1]
        rec_img=Image.fromarray(rec_rgb,"RGB").resize((Config.HALF_RECORDING_WIDTH, Config.HALF_RECORDING_HEIGHT), Image.LANCZOS)

        emi_t=torch.from_numpy(np.array(emi_img).astype(np.float32)/255.0).permute(2,0,1)
        rec_t=torch.from_numpy(np.array(rec_img).astype(np.float32)/255.0).permute(2,0,1)
        return emi_t, rec_t

    def close(self):
        for f,h,n in self.datasets:
            h.close()

def pad_and_center(emi, rec):
    # Pad both images so that they are the same size and both are centered.
    B,C_e,H_e,W_e=emi.shape
    B,C_r,H_r,W_r=rec.shape
    Ht=max(H_e,H_r)
    Wt=max(W_e,W_r)

    # For emission:
    dH_e=Ht-H_e
    top_e=dH_e//2
    bottom_e=dH_e - top_e
    dW_e=Wt-W_e
    left_e=dW_e//2
    right_e=dW_e - left_e
    emi_p=F.pad(emi,(left_e,right_e,top_e,bottom_e))

    # For recording:
    dH_r=Ht-H_r
    top_r=dH_r//2
    bottom_r=dH_r - top_r
    dW_r=Wt-W_r
    left_r=dW_r//2
    right_r=dW_r - left_r
    rec_p=F.pad(rec,(left_r,right_r,top_r,bottom_r))

    return emi_p, rec_p

def train_autoencoder():
    files=glob.glob(os.path.join(Config.OUTPUT_DIR,"*","data.h5"))
    if len(files)==0:
        print("No data found. Run main_record.py first.")
        return
    logger.info(f"Found {len(files)} datasets for AE training.")

    ds=MultiHDF5Dataset(files)
    dl=DataLoader(ds,batch_size=Config.BATCH_SIZE,shuffle=True,num_workers=4)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae=AdversarialAutoencoder().to(device)
    disc=Discriminator(in_ch=6).to(device)

    opt_g=torch.optim.Adam(ae.parameters(), lr=Config.LEARNING_RATE, betas=(0.5,0.999))
    opt_d=torch.optim.Adam(disc.parameters(), lr=Config.LEARNING_RATE, betas=(0.5,0.999))

    bce=nn.BCEWithLogitsLoss()
    l1=nn.L1Loss()

    vgg=models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:36].eval().to(device)
    for p in vgg.parameters():
        p.requires_grad=False
    def vgg_loss(x,y):
        mean=torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1)
        std=torch.tensor([0.229,0.224,0.225],device=device).view(1,3,1,1)
        x_norm=(x-mean)/std
        y_norm=(y-mean)/std
        fx=vgg(x_norm)
        fy=vgg(y_norm)
        return F.l1_loss(fx,fy)

    create_dir_if_not_exists(Config.TRAINING_OUTPUTS_DIR)

    scaler=GradScaler()

    start_epoch=0
    if os.path.isfile("aae_model_latest.pth") and os.path.isfile("aae_discriminator_latest.pth"):
        ae.load_state_dict(torch.load("aae_model_latest.pth", map_location=device))
        disc.load_state_dict(torch.load("aae_discriminator_latest.pth", map_location=device))
        if os.path.isfile("aae_epoch.txt"):
            start_epoch=int(open("aae_epoch.txt").read().strip())

    ADV_WEIGHT=Config.ADV_WEIGHT
    RECON_WEIGHT=Config.RECON_WEIGHT
    VGG_WEIGHT=0.2

    # Increase EPOCHS to 200
    AE_EPOCHS=200

    steps_per_epoch=len(dl)
    for epoch in range(start_epoch, AE_EPOCHS):
        ae.train()
        disc.train()
        epoch_g_loss=0.0
        epoch_d_loss=0.0
        start_time=time.time()

        for i,(emi,rec) in enumerate(dl):
            emi=emi.to(device)
            rec=rec.to(device)
            emi_p, rec_p = pad_and_center(emi, rec)

            opt_g.zero_grad()
            with autocast():
                re, rr=ae(emi_p,rec_p)
                inp_fake=torch.cat([re, rr],1)
                pred_fake=disc(inp_fake)
                lbl_real=torch.ones_like(pred_fake,device=device)*Config.REAL_LABEL
                loss_g_adv=ADV_WEIGHT*bce(pred_fake,lbl_real)

                loss_recon=(l1(re,emi_p)+l1(rr,rec_p))*0.5*RECON_WEIGHT

                target_size=(256,256)
                emi_v=F.interpolate(emi_p,size=target_size,mode='bilinear',align_corners=False)
                rec_v=F.interpolate(rec_p,size=target_size,mode='bilinear',align_corners=False)
                re_v=F.interpolate(re,size=target_size,mode='bilinear',align_corners=False)
                rr_v=F.interpolate(rr,size=target_size,mode='bilinear',align_corners=False)
                vgg_loss_emi=vgg_loss(emi_v,re_v)
                vgg_loss_rec=vgg_loss(rec_v,rr_v)
                loss_vgg=(vgg_loss_emi+vgg_loss_rec)*0.5*VGG_WEIGHT

                loss_g=loss_g_adv+loss_recon+loss_vgg

            scaler.scale(loss_g).backward()
            scaler.step(opt_g)
            opt_g.zero_grad()

            opt_d.zero_grad()
            with autocast():
                inp_real=torch.cat([emi_p,rec_p],1)
                pred_real=disc(inp_real)
                lbl_real2=torch.ones_like(pred_real,device=device)*Config.REAL_LABEL
                lbl_fake2=torch.zeros_like(pred_real,device=device)*Config.FAKE_LABEL

                pred_fake_det=disc(inp_fake.detach())
                loss_d_real=bce(pred_real,lbl_real2)
                loss_d_fake=bce(pred_fake_det,lbl_fake2)
                loss_d_total=(loss_d_real+loss_d_fake)*0.5

            scaler.scale(loss_d_total).backward()
            scaler.step(opt_d)
            opt_d.zero_grad()

            scaler.update()

            epoch_g_loss+=loss_g.item()
            epoch_d_loss+=loss_d_total.item()

            if (i+1)%100==0:
                l1_val=((re-emi_p).abs().mean().item()+(rr-rec_p).abs().mean().item())*0.5
                logger.info(f"AE Epoch {epoch} Step {i+1}/{steps_per_epoch}: G={loss_g.item():.4f}, D={loss_d_total.item():.4f}, L1={l1_val:.4f}")

        epoch_g_loss/=steps_per_epoch
        epoch_d_loss/=steps_per_epoch
        elapsed=time.time()-start_time
        logger.info(f"AE Epoch {epoch}: G={epoch_g_loss:.4f}, D={epoch_d_loss:.4f}, time={elapsed:.2f}s")

        if (epoch+1)%Config.MODEL_SAVE_EVERY==0:
            torch.save(ae.state_dict(),"aae_model_latest.pth")
            torch.save(disc.state_dict(),"aae_discriminator_latest.pth")
            with open("aae_epoch.txt","w") as f:
                f.write(str(epoch+1))

        if (epoch+1)%Config.SAMPLE_OUTPUT_EVERY==0:
            ae.eval()
            with torch.no_grad():
                emi_s,rec_s=next(iter(dl))
                emi_s,rec_s=emi_s.to(device),rec_s.to(device)
                emi_ps,rec_ps=pad_and_center(emi_s,rec_s)
                re_s, rr_s=ae(emi_ps,rec_ps)
                re_img=tensor_to_npimg(re_s[0])
                rr_img=tensor_to_npimg(rr_s[0])
                create_dir_if_not_exists(Config.TRAINING_OUTPUTS_DIR)
                Image.fromarray(re_img,"RGB").save(os.path.join(Config.TRAINING_OUTPUTS_DIR,f"epoch_{epoch+1}_emi.png"))
                Image.fromarray(rr_img,"RGB").save(os.path.join(Config.TRAINING_OUTPUTS_DIR,f"epoch_{epoch+1}_rec.png"))

    ds.close()
    logger.info("AE Training complete.")

def train_latent_gan():
    files=glob.glob(os.path.join(Config.OUTPUT_DIR,"*","data.h5"))
    if len(files)==0:
        print("No data for latent GAN.")
        return
    ds=MultiHDF5Dataset(files)
    dl=DataLoader(ds,batch_size=Config.LATENT_GAN_BATCH_SIZE,shuffle=True,num_workers=4)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile("aae_model_latest.pth"):
        print("No trained AE model found for latent GAN. Train AE first.")
        return

    ae=AdversarialAutoencoder().eval().to(device)
    ae.load_state_dict(torch.load("aae_model_latest.pth", map_location=device))
    for p in ae.parameters():
        p.requires_grad=False

    # Test shape from one batch
    emi,rec=next(iter(dl))
    emi,rec=emi.to(device),rec.to(device)
    emi_p, rec_p = pad_and_center(emi, rec)
    with torch.no_grad():
        ze=ae.emission_encoder(emi_p)
        zr=ae.recording_encoder(rec_p)
        fused=ae.forward_latent(ze,zr)
        fused_64=F.adaptive_avg_pool2d(fused,(64,64))
    in_ch=fused_64.shape[1]

    disc_latent=LatentDiscriminator(in_ch=in_ch).to(device)
    gen_latent=LatentGenerator(noise_dim=Config.NOISE_DIM,out_ch=in_ch).to(device)

    opt_g=torch.optim.Adam(gen_latent.parameters(), lr=Config.LATENT_GAN_LR, betas=(0.5,0.999))
    opt_d=torch.optim.Adam(disc_latent.parameters(), lr=Config.LATENT_GAN_LR, betas=(0.5,0.999))
    bce=nn.BCEWithLogitsLoss()

    scaler=GradScaler()

    start_epoch=0
    if os.path.isfile("latent_gan_generator_latest.pth") and os.path.isfile("latent_gan_discriminator_latest.pth"):
        gen_latent.load_state_dict(torch.load("latent_gan_generator_latest.pth",map_location=device))
        disc_latent.load_state_dict(torch.load("latent_gan_discriminator_latest.pth",map_location=device))
        if os.path.isfile("latent_gan_epoch.txt"):
            start_epoch=int(open("latent_gan_epoch.txt").read().strip())

    # Train latent GAN for 200 epochs
    LATENT_GAN_EPOCHS=200
    steps_per_epoch=len(dl)

    for epoch in range(start_epoch, LATENT_GAN_EPOCHS):
        disc_latent.train()
        gen_latent.train()
        epoch_d_loss=0
        epoch_g_loss=0
        start_time=time.time()
        for i,(emi,rec) in enumerate(dl):
            emi,rec=emi.to(device),rec.to(device)
            emi_p, rec_p=pad_and_center(emi,rec)
            with torch.no_grad():
                ze=ae.emission_encoder(emi_p)
                zr=ae.recording_encoder(rec_p)
                fused=ae.forward_latent(ze,zr)
                fused_64=F.adaptive_avg_pool2d(fused,(64,64))
            B=fused_64.shape[0]

            noise=torch.randn(B,Config.NOISE_DIM,device=device)
            with autocast():
                fake_latent=gen_latent(noise)
                pred_real=disc_latent(fused_64)
                pred_fake=disc_latent(fake_latent.detach())
                lbl_real=torch.ones_like(pred_real,device=device)*Config.REAL_LABEL
                lbl_fake=torch.zeros_like(pred_fake,device=device)*Config.FAKE_LABEL
                loss_d=(bce(pred_real,lbl_real)+bce(pred_fake,lbl_fake))*0.5

            opt_d.zero_grad()
            scaler.scale(loss_d).backward()
            scaler.step(opt_d)
            opt_d.zero_grad(set_to_none=True)

            with autocast():
                pred_fake_g=disc_latent(fake_latent)
                loss_g=bce(pred_fake_g,lbl_real)

            opt_g.zero_grad()
            scaler.scale(loss_g).backward()
            scaler.step(opt_g)
            scaler.update()
            opt_g.zero_grad(set_to_none=True)

            epoch_d_loss+=loss_d.item()
            epoch_g_loss+=loss_g.item()

            if (i+1)%50==0:
                logger.info(f"LatentGAN Epoch {epoch} Step {i+1}/{steps_per_epoch}: D={loss_d.item():.4f} G={loss_g.item():.4f}")

        epoch_d_loss/=steps_per_epoch
        epoch_g_loss/=steps_per_epoch
        elapsed=time.time()-start_time
        logger.info(f"LatentGAN Epoch {epoch}: D={epoch_d_loss:.4f}, G={epoch_g_loss:.4f}, time={elapsed:.2f}s")

        if (epoch+1)%Config.LATENT_GAN_MODEL_SAVE_EVERY==0:
            torch.save(gen_latent.state_dict(),"latent_gan_generator_latest.pth")
            torch.save(disc_latent.state_dict(),"latent_gan_discriminator_latest.pth")
            with open("latent_gan_epoch.txt","w") as f:
                f.write(str(epoch+1))

        if (epoch+1)%Config.LATENT_GAN_SAMPLE_OUTPUT_EVERY==0:
            gen_latent.eval()
            with torch.no_grad():
                noise=torch.randn(4,Config.NOISE_DIM,device=device)
                fake_latent=gen_latent(noise)
                emi_fake=ae.emission_decoder(fake_latent)
                rec_fake=ae.recording_decoder(fake_latent)
            create_dir_if_not_exists(Config.SAMPLES_DIR)
            for j in range(4):
                re_img=tensor_to_npimg(emi_fake[j])
                rr_img=tensor_to_npimg(rec_fake[j])
                Image.fromarray(re_img,"RGB").save(os.path.join(Config.SAMPLES_DIR,f"latent_gan_epoch_{epoch+1}_emi_{j}.png"))
                Image.fromarray(rr_img,"RGB").save(os.path.join(Config.SAMPLES_DIR,f"latent_gan_epoch_{epoch+1}_rec_{j}.png"))

    ds.close()
    logger.info("Latent GAN training complete.")

def main():
    train_autoencoder()
    train_latent_gan()

if __name__=="__main__":
    main()

