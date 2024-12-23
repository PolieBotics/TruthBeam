import torch
import torch.nn as nn
import torch.nn.functional as F

########################################
# Simple Residual Block
########################################

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self,x):
        r=x
        x=self.conv1(x)
        x=self.bn1(x)
        x=F.relu(x, inplace=True)
        x=self.conv2(x)
        x=self.bn2(x)
        return F.relu(x+r, inplace=True)

########################################
# Encoder / Decoder for AAE
########################################

class Encoder(nn.Module):
    def __init__(self, in_ch=3, base=32, levels=2):
        super().__init__()
        ch=base
        layers = [nn.Conv2d(in_ch, ch, 3,1,1), nn.ReLU(True)]
        for _ in range(levels):
            layers.append(nn.Conv2d(ch, ch*2,4,2,1))
            ch*=2
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU(True))
        layers.append(ResidualBlock(ch))
        self.net=nn.Sequential(*layers)
        self.out_ch=ch
    def forward(self,x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, out_ch=3, base=32, levels=2):
        super().__init__()
        ch=base*(2**levels)
        layers=[ResidualBlock(ch)]
        for _ in range(levels):
            layers.append(nn.ConvTranspose2d(ch,ch//2,4,2,1))
            ch//=2
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(ch,out_ch,3,1,1))
        self.net=nn.Sequential(*layers)
    def forward(self,x):
        return torch.sigmoid(self.net(x))

########################################
# Adversarial Autoencoder
########################################

class AdversarialAutoencoder(nn.Module):
    def __init__(self, base=32, levels=2):
        super().__init__()
        self.emission_encoder = Encoder(3, base, levels)
        self.recording_encoder= Encoder(3, base, levels)
        ch=self.emission_encoder.out_ch*2
        self.fusion=ResidualBlock(ch)
        self.post_fusion=nn.Conv2d(ch,self.emission_encoder.out_ch,1,1)
        self.emission_decoder=Decoder(3,base,levels)
        self.recording_decoder=Decoder(3,base,levels)

    def forward_latent(self,ze,zr):
        fused_in=torch.cat([ze,zr],1)
        fused=self.fusion(fused_in)
        fused=self.post_fusion(fused)
        return fused

    def forward(self,emi,rec):
        ze=self.emission_encoder(emi)
        zr=self.recording_encoder(rec)
        fused=self.forward_latent(ze,zr)
        re=self.emission_decoder(fused)
        rr=self.recording_decoder(fused)
        return re, rr

########################################
# Image Discriminator
########################################

class Discriminator(nn.Module):
    def __init__(self, in_ch=6, base=32):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_ch, base,4,2,1), nn.LeakyReLU(0.2,True),
            nn.Conv2d(base,base*2,4,2,1), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2,True),
            nn.Conv2d(base*2,base*4,4,2,1), nn.BatchNorm2d(base*4), nn.LeakyReLU(0.2,True),
            nn.Conv2d(base*4,1,4,1,1)
        )
    def forward(self,x):
        return self.net(x)

########################################
# Latent Discriminator and Generator
########################################

class LatentDiscriminator(nn.Module):
    def __init__(self, in_ch=128):
        super().__init__()
        base=32
        self.net=nn.Sequential(
            nn.Conv2d(in_ch, base,4,2,1), nn.LeakyReLU(0.2,True),
            nn.Conv2d(base, base*2,4,2,1), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2,True),
            nn.Conv2d(base*2, base*4,4,2,1), nn.BatchNorm2d(base*4), nn.LeakyReLU(0.2,True),
            nn.Conv2d(base*4,1,4,1,1)
        )
    def forward(self,x):
        return self.net(x)

class LatentGenerator(nn.Module):
    def __init__(self, noise_dim=128, out_ch=128):
        super().__init__()
        base=256
        layers=[]
        layers.append(nn.ConvTranspose2d(noise_dim,base,4,1,0))
        layers.append(nn.BatchNorm2d(base))
        layers.append(nn.ReLU(True))
        layers.append(nn.ConvTranspose2d(base,base//2,4,2,1))
        base//=2
        layers.append(nn.BatchNorm2d(base))
        layers.append(nn.ReLU(True))
        layers.append(nn.ConvTranspose2d(base,base//2,4,2,1))
        base//=2
        layers.append(nn.BatchNorm2d(base))
        layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(base,out_ch,3,1,1))
        self.net=nn.Sequential(*layers)

    def forward(self,z):
        z=z.unsqueeze(-1).unsqueeze(-1)
        x=self.net(z)
        x=F.adaptive_avg_pool2d(x,(64,64))
        return x

