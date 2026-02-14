import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import torch
import math
import numpy as np
from torchvision.models import resnet50

basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 512
num_heads = 8
depth = 1
num_modals = 3  # RGB, NIR ve SWIR
patch_size = 8

def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError(f"normalization type {norm} is not supported")
    return m
   
class general_conv3d_prenorm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', act_type='relu', relufactor=0.2):
        super(general_conv3d_prenorm, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.norm(x)
        return x  

class fusion_prenorm(nn.Module):
    def __init__(self, in_channel=64, num_cls=1):
        super(fusion_prenorm, self).__init__()
        self.fusion_layer = nn.Sequential(
                        general_conv3d_prenorm(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d_prenorm(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d_prenorm(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x):
        return self.fusion_layer(x)

# -----------------------------------------------------------
# EarlyFusionBlock: Üç modalitenin özelliklerini
# concat edip 1x1 konvolüsyon ile decoder'ın beklediği
# kanal sayısını sağlayacak şekilde (örneğin x1 için 3*basic_dims)
# dönüştürüyoruz.
# -----------------------------------------------------------
class EarlyFusionBlock(nn.Module):
    def __init__(self, in_channels):
        """
        :param in_channels: Her modality'nin encoder çıktısının kanal sayısı.
                             Çıkışta concatenation sonucu 3*in_channels elde edilecektir.
        """
        super(EarlyFusionBlock, self).__init__()
        total_ch = num_modals * in_channels  # 3 * in_channels
        self.conv = nn.Conv3d(total_ch, total_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.norm = normalization(total_ch, norm='in')

    def forward(self, x_rgb, x_nir, x_swir):
        x = torch.cat([x_rgb, x_nir, x_swir], dim=1)
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        return x

def inflate_conv(conv2d, time_dim):
    """
    conv2d: nn.Conv2d
    time_dim:  3D’deki temporal kernel boyutu (ör: 3)
    """
    # 3D conv katmanını inşa et
    conv3d = nn.Conv3d(
        in_channels = 1 if conv2d.in_channels == 3 else conv2d.in_channels,
        out_channels= conv2d.out_channels,
        kernel_size = (time_dim, *conv2d.kernel_size),
        stride      = (1, *conv2d.stride),
        padding     = (time_dim//2, *conv2d.padding),
        bias        = conv2d.bias is not None
    )
    # 2D ağırlığı al
    w2d = conv2d.weight.data            # [O, I, K, K]
    # Eğer I==3 ve 3D in_channels==1 ise, kanalların ortalamasını alın
    if w2d.shape[1] == 3 and conv3d.weight.shape[1] == 1:
        # [O,3,K,K] → [O,1,1,K,K] → yayarak [O,1,T,K,K]
        w3d = w2d.mean(dim=1, keepdim=True).unsqueeze(2).repeat(1,1,time_dim,1,1)
    else:
        # diğer durumlarda basit repeat
        w3d = w2d.unsqueeze(2).repeat(1,1,time_dim,1,1) / time_dim

    # kopyala
    conv3d.weight.data.copy_(w3d)
    if conv2d.bias is not None:
        conv3d.bias.data.copy_(conv2d.bias.data)
    return conv3d

class Encoder(nn.Module):
    def __init__(self, inflate_time=3):
        super().__init__()
        # 1) 2D-ResNet50’ten katmanları al
        res2d = resnet50(pretrained=True)

        # 2) conv1 + bn1 + relu + maxpool  → e1
        self.e1_c1 = inflate_conv(res2d.conv1, inflate_time)
        self.e1_bn  = nn.BatchNorm3d(res2d.bn1.num_features)
        self.e1_relu= res2d.relu
        self.e1_mp  = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        # 3) layer1…layer4’ü inflate ederek ardışık bloklara dönüştür
        def inflate_layer(layer2d):
            layer3d = []
            for bottleneck in layer2d:
                # her bottleneck içindeki conv1,conv2,conv3 + bn’leri inflate et
                b = bottleneck
                conv1 = inflate_conv(b.conv1, time_dim=1)
                bn1   = nn.BatchNorm3d(b.bn1.num_features)
                conv2 = inflate_conv(b.conv2, time_dim=1)
                bn2   = nn.BatchNorm3d(b.bn2.num_features)
                conv3 = inflate_conv(b.conv3, time_dim=1)
                bn3   = nn.BatchNorm3d(b.bn3.num_features)
                down = None
                if b.downsample is not None:
                    # downsample: conv + bn
                    ds_conv2d = b.downsample[0]
                    ds_bn2d   = b.downsample[1]
                    ds_conv3d = inflate_conv(ds_conv2d, time_dim=1)
                    ds_bn3d   = nn.BatchNorm3d(ds_bn2d.num_features)
                    down = nn.Sequential(ds_conv3d, ds_bn3d)
                # birleşik 3D-Bottleneck
                layer3d.append(
                    Bottleneck3D(
                        conv1, bn1, conv2, bn2, conv3, bn3, down
                    )
                )
            return nn.Sequential(*layer3d)

        # 4) 2D layer1–4’ü alıp 3D’ye dönüştür
        self.e2 = inflate_layer(res2d.layer1)
        self.e3 = inflate_layer(res2d.layer2)
        self.e4 = inflate_layer(res2d.layer3)
        self.e5 = inflate_layer(res2d.layer4)

        # 5) x6 üretimi için orijinal conv (basic_dims*1+2+4+8+8 → basic_dims*8)
        c = basic_dims*(1+2+4+8+8)
        self.conv6 = nn.Conv3d(in_channels=c, out_channels=basic_dims*8, kernel_size=1)

        # 6) Son olarak her seviye kanallarını basic_dims tabanına düşürelim:
        self.adapt1 = nn.Conv3d(64,  basic_dims,    kernel_size=1)
        self.adapt2 = nn.Conv3d(256, basic_dims*2,  kernel_size=1)
        self.adapt3 = nn.Conv3d(512, basic_dims*4,  kernel_size=1)
        self.adapt4 = nn.Conv3d(1024, basic_dims*8, kernel_size=1)
        self.adapt5 = nn.Conv3d(2048, basic_dims*8, kernel_size=1)

    def forward(self, x):
        # e1
        x1 = self.e1_c1(x)
        x1 = self.e1_bn(self.e1_relu(x1))
        x1 = self.e1_mp(x1)
        # e2–e5
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.e5(x4)
        # adapt kanal sayıları
        x1 = self.adapt1(x1)
        x2 = self.adapt2(x2)
        x3 = self.adapt3(x3)
        x4 = self.adapt4(x4)
        x5 = self.adapt5(x5)
        # x6: aynı orijinal kod gibi interpolate + concat + conv6
        x1_ = F.interpolate(x1, size=(8,8,8), mode='trilinear', align_corners=True)
        x2_ = F.interpolate(x2, size=(8,8,8), mode='trilinear', align_corners=True)
        x3_ = F.interpolate(x3, size=(8,8,8), mode='trilinear', align_corners=True)
        x4_ = F.interpolate(x4, size=(8,8,8), mode='trilinear', align_corners=True)
        x5_ = F.interpolate(x5, size=(8,8,8), mode='trilinear', align_corners=True)
        x6 = torch.cat([x1_,x2_,x3_,x4_,x5_], dim=1)
        x6 = self.conv6(x6)
        return x1, x2, x3, x4, x5, x6
        
class Bottleneck3D(nn.Module):
    def __init__(self, c1,b1, c2,b2, c3,b3, down=None):
        super().__init__()
        self.conv1, self.bn1 = c1,b1
        self.conv2, self.bn2 = c2,b2
        self.conv3, self.bn3 = c3,b3
        self.relu = nn.ReLU(inplace=True)
        self.downsample = down
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class sigmoidOut(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(sigmoidOut, self).__init__()
        self.outsig = nn.Sigmoid()
    def forward(self, x):
        return self.outsig(x)

# Decoder_fuse aynı şekilde bırakılıyor (hiçbir değişiklik yapılmıyor)
class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=1):
        super(Decoder_fuse, self).__init__()
        self.d4_c1 = general_conv3d_prenorm(basic_dims*16, basic_dims*16, pad_type='replicate')
        self.d4_c2 = general_conv3d_prenorm(320, basic_dims*8, pad_type='replicate')
        self.d4_out = general_conv3d_prenorm(basic_dims*8, basic_dims*8, k_size=1, padding=0, pad_type='replicate')
        self.d3_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='replicate')
        self.d3_c2 = general_conv3d_prenorm(128, basic_dims*4, pad_type='replicate')
        self.d3_out = general_conv3d_prenorm(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='replicate')
        self.d2_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='replicate')
        self.d2_c2 = general_conv3d_prenorm(64, basic_dims*2, pad_type='replicate')
        self.d2_out = general_conv3d_prenorm(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='replicate')
        self.d1_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='replicate')
        self.d1_c2 = general_conv3d_prenorm(32, basic_dims, pad_type='replicate')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='replicate')
        self.seg_d4 = nn.Conv3d(in_channels=basic_dims*8, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=basic_dims*8, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=basic_dims*4, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=basic_dims*2, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.outsig = sigmoidOut(1, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)
       
       
        self.RFM5 = fusion_prenorm(in_channel=basic_dims*8*3, num_cls=1)

        self.RFM5_reduce = nn.Conv3d(
            in_channels=basic_dims*8*3,   # 8*8*3 = 192
            out_channels=basic_dims*16,   # 8*16   = 128
            kernel_size=1, stride=1, padding=0, bias=True
        )
       

        self.RFM4 = fusion_prenorm(in_channel=basic_dims*8*3, num_cls=1)
        self.RFM3 = fusion_prenorm(in_channel=basic_dims*4*3, num_cls=1)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims*2*3, num_cls=1)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims*1*3, num_cls=1)

        self.up_to_224 = nn.Upsample(size=(1, 224, 224), mode='trilinear', align_corners=True)
        self.final_conv = nn.Conv3d(in_channels=8, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.RFM5(x5)
        de_x5 = self.RFM5_reduce(de_x5)
        de_x5 = self.d4_c1(self.up2(de_x5))
        de_x4 = self.RFM4(x4)
        de_x4 = F.interpolate(de_x4, (16, 16, 16))
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))
        de_x3 = self.RFM3(x3)
        de_x3 = F.interpolate(de_x3, (32, 32, 32))
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))
        de_x2 = self.RFM2(x2)
        de_x2 = F.interpolate(de_x2, (64, 64, 64))
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))
        de_x1 = self.RFM1(x1)
        de_x1 = F.interpolate(de_x1, (128, 128, 128))
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))
        de_x1_up = self.up_to_224(de_x1)
        logits = self.final_conv(de_x1_up)
        pred = self.outsig(logits)
        return pred


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x):
        return self.fn(self.norm(x))

class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn
    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_levels=1, n_points=4):
        super(Transformer, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )
        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)
    def forward(self, x, pos):
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x


class MMVit4(nn.Module):
    def __init__(self, num_cls=1):
        super(MMVit4, self).__init__()
        self.RGB_encoder = Encoder()
        self.NIR_encoder = Encoder()
        self.SWIR_encoder = Encoder()

        self.RGB_encode_conv = nn.Conv3d(basic_dims*8, transformer_basic_dims, 1)
        self.NIR_encode_conv = nn.Conv3d(basic_dims*8, transformer_basic_dims, 1)
        self.SWIR_encode_conv = nn.Conv3d(basic_dims*8, transformer_basic_dims, 1)

        self.fused6_encode_conv = nn.Conv3d(basic_dims * 8 * 3, transformer_basic_dims, 1)

        self.RGB_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*8, 1)
        self.NIR_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*8, 1)
        self.SWIR_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*8, 1)

        self.RGB_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.NIR_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.SWIR_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.fused6_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))

        self.RGB_transformer = Transformer(transformer_basic_dims, depth, num_heads, mlp_dim)
        self.NIR_transformer = Transformer(transformer_basic_dims, depth, num_heads, mlp_dim)
        self.SWIR_transformer = Transformer(transformer_basic_dims, depth, num_heads, mlp_dim)

        self.qkv_RGB = nn.Conv3d(transformer_basic_dims, transformer_basic_dims * 3, 1)
        self.qkv_NIR = nn.Conv3d(transformer_basic_dims, transformer_basic_dims * 3, 1)
        self.qkv_SWIR = nn.Conv3d(transformer_basic_dims, transformer_basic_dims * 3, 1)

        self.softmax_RGB = nn.Softmax(dim=0)
        self.softmax_NIR = nn.Softmax(dim=0)
        self.softmax_SWIR = nn.Softmax(dim=0)

        self.multimodal_transformer = Transformer(transformer_basic_dims, depth, num_heads, mlp_dim, n_levels=3)
        self.multimodal_decode_conv = nn.Conv3d(transformer_basic_dims * 4, basic_dims * 8 * 3, 1)

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)


        self.fusion1 = EarlyFusionBlock(basic_dims)
        self.fusion2 = EarlyFusionBlock(basic_dims * 2)
        self.fusion3 = EarlyFusionBlock(basic_dims * 4)
        self.fusion4 = EarlyFusionBlock(basic_dims * 8)
        self.fusion5 = EarlyFusionBlock(basic_dims * 8)
        self.fusion6 = EarlyFusionBlock(basic_dims * 8)

        # -------------------------------------------------
        # SHARED + PRIVATE LATENT PROJEKSİYONLARI
        #   shared:  basic_dims*12  = 96 kanal
        #   private: her modality için basic_dims*4 = 32 kanal
        #   Toplam:  96 + 3*32 = 192 = basic_dims*8*3 (decoder ile uyumlu)
        # -------------------------------------------------
        shared_channels = basic_dims * 12   # 96
        private_channels = basic_dims * 4   # 32

        # Shared latent: x6_inter (192 kanal) -> 96 kanal
        self.shared_conv = nn.Conv3d(
            in_channels=basic_dims * 8 * 3,   # 192
            out_channels=shared_channels,     # 96
            kernel_size=1, stride=1, padding=0, bias=True
        )

        # Private latent'ler: x6_RGB_/NIR_/SWIR_ (512 kanal) -> 32 kanal
        self.rgb_private_conv = nn.Conv3d(
            in_channels=transformer_basic_dims,   # 512
            out_channels=private_channels,        # 32
            kernel_size=1, stride=1, padding=0, bias=True
        )
        self.nir_private_conv = nn.Conv3d(
            in_channels=transformer_basic_dims,
            out_channels=private_channels,
            kernel_size=1, stride=1, padding=0, bias=True
        )
        self.swir_private_conv = nn.Conv3d(
            in_channels=transformer_basic_dims,
            out_channels=private_channels,
            kernel_size=1, stride=1, padding=0, bias=True
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

    def _apply_modality_mask(self, feats, mask_col):
        """
        feats: encoder çıktılarının listesi [x1, x2, x3, x4, x5, x6]
               her biri [B, C, D, H, W]
        mask_col: [B] şeklinde, bu modalitenin var/yok bilgisi (0/1 veya bool)
        """
        if mask_col is None:
            return feats

        B = feats[0].size(0)
        m = mask_col.view(B, 1, 1, 1, 1).to(feats[0].dtype).to(feats[0].device)
        return [f * m for f in feats]

    def forward(self, x, mask=None):
        """
        x: [B, 3, D, H, W]  (RGB, NIR, SWIR)
        mask: [B, 3] veya None
              mask[b,0] = 0 → o batch’te RGB eksik
              mask[b,1] = 0 → NIR eksik
              mask[b,2] = 0 → SWIR eksik
              None → tüm modaliteler mevcut (full-case; full decoder kullanılır)
        """
        B = x.size(0)

        # --- 1) Encoder çıktıları ---
        RGB_x1, RGB_x2, RGB_x3, RGB_x4, RGB_x5, RGB_x6 = \
            self.RGB_encoder(x[:, 0:1, :, :, :])
        NIR_x1, NIR_x2, NIR_x3, NIR_x4, NIR_x5, NIR_x6 = \
            self.NIR_encoder(x[:, 1:2, :, :, :])
        SWIR_x1, SWIR_x2, SWIR_x3, SWIR_x4, SWIR_x5, SWIR_x6 = \
            self.SWIR_encoder(x[:, 2:3, :, :, :])

        # --- 2) Mask ile encoder seviyesinde gating ---
        if mask is not None:
            rgb_mask  = mask[:, 0]
            nir_mask  = mask[:, 1]
            swir_mask = mask[:, 2]

            RGB_x1, RGB_x2, RGB_x3, RGB_x4, RGB_x5, RGB_x6 = \
                self._apply_modality_mask([RGB_x1, RGB_x2, RGB_x3, RGB_x4, RGB_x5, RGB_x6],
                                          rgb_mask)
            NIR_x1, NIR_x2, NIR_x3, NIR_x4, NIR_x5, NIR_x6 = \
                self._apply_modality_mask([NIR_x1, NIR_x2, NIR_x3, NIR_x4, NIR_x5, NIR_x6],
                                          nir_mask)
            SWIR_x1, SWIR_x2, SWIR_x3, SWIR_x4, SWIR_x5, SWIR_x6 = \
                self._apply_modality_mask([SWIR_x1, SWIR_x2, SWIR_x3, SWIR_x4, SWIR_x5, SWIR_x6],
                                          swir_mask)

        # --- 3) Early fusion ---
        fused_x1 = self.fusion1(RGB_x1, NIR_x1, SWIR_x1)
        fused_x2 = self.fusion2(RGB_x2, NIR_x2, SWIR_x2)
        fused_x3 = self.fusion3(RGB_x3, NIR_x3, SWIR_x3)
        fused_x4 = self.fusion4(RGB_x4, NIR_x4, SWIR_x4)
        fused_x5 = self.fusion5(RGB_x5, NIR_x5, SWIR_x5)
        fused_x6 = self.fusion6(RGB_x6, NIR_x6, SWIR_x6)

        # --- 4) IntraFormer tokenizasyonu ---
        def tokenize(x6, conv, pos, transformer):
            token = conv(x6) \
                .permute(0, 2, 3, 4, 1) \
                .contiguous() \
                .view(B, -1, transformer_basic_dims)
            return transformer(token, pos), token

        RGB_trans, RGB_skip = tokenize(RGB_x6, self.RGB_encode_conv, self.RGB_pos, self.RGB_transformer)
        NIR_trans, NIR_skip = tokenize(NIR_x6, self.NIR_encode_conv, self.NIR_pos, self.NIR_transformer)
        SWIR_trans, SWIR_skip = tokenize(SWIR_x6, self.SWIR_encode_conv, self.SWIR_pos, self.SWIR_transformer)

        # --- 5) InterFormer (multimodal correlation) ---
        def qkv(x, layer):
            return layer(x).chunk(3, dim=1)

        def reshape_for_qkv(t):
            return t.view(B, patch_size, patch_size, patch_size, -1) \
                    .permute(0, 4, 1, 2, 3)

        q_r, k_r, v_r = qkv(reshape_for_qkv(RGB_trans), self.qkv_RGB)
        q_n, k_n, v_n = qkv(reshape_for_qkv(NIR_trans), self.qkv_NIR)
        q_s, k_s, v_s = qkv(reshape_for_qkv(SWIR_trans), self.qkv_SWIR)

        def inter_attn(q, ks, vs, softmax_fn):
            scores = [q * k for k in ks]
            concat = torch.cat([s.reshape(1, -1) for s in scores], dim=0)
            attn = softmax_fn(concat / math.sqrt(len(ks)))
            attn = attn.view(q.size(0), q.size(1)*len(ks), q.size(2), q.size(3), q.size(4))
            return sum(attn[:, i*q.size(1):(i+1)*q.size(1)] * v
                       for i, v in enumerate(vs))

        x6_RGB_  = inter_attn(q_r, [k_r, k_n, k_s], [v_r, v_n, v_s], self.softmax_RGB)
        x6_NIR_  = inter_attn(q_n, [k_r, k_n, k_s], [v_r, v_n, v_s], self.softmax_NIR)
        x6_SWIR_ = inter_attn(q_s, [k_r, k_n, k_s], [v_r, v_n, v_s], self.softmax_SWIR)

        # (Opsiyonel ama tutarlı): burada da maskeyi yeniden uygula
        if mask is not None:
            rgb_mask  = mask[:, 0].view(B, 1, 1, 1, 1).to(x6_RGB_.dtype).to(x6_RGB_.device)
            nir_mask  = mask[:, 1].view(B, 1, 1, 1, 1).to(x6_NIR_.dtype).to(x6_NIR_.device)
            swir_mask = mask[:, 2].view(B, 1, 1, 1, 1).to(x6_SWIR_.dtype).to(x6_SWIR_.device)
            x6_RGB_  = x6_RGB_  * rgb_mask
            x6_NIR_  = x6_NIR_  * nir_mask
            x6_SWIR_ = x6_SWIR_ * swir_mask

        # --- 6) Intra-former çıktılarının birleştirilmesi ---
        x6_intra = torch.stack((x6_RGB_, x6_NIR_, x6_SWIR_), dim=1) \
                        .view(B, -1, patch_size, patch_size, patch_size)

        RGB_corr, NIR_corr, SWIR_corr = [
            part.permute(0,2,3,4,1)
                .contiguous()
                .view(B, -1, transformer_basic_dims)
            for part in torch.chunk(x6_intra, 3, dim=1)
        ]

        RGB_fused = RGB_skip + RGB_corr
        NIR_fused = NIR_skip + NIR_corr
        SWIR_fused = SWIR_skip + SWIR_corr

        # --- 7) Multimodal token oluşturma ---
        fused6_token = self.fused6_encode_conv(fused_x6) \
            .permute(0,2,3,4,1) \
            .contiguous() \
            .view(B, -1, transformer_basic_dims)

        multimodal_tokens = torch.cat([RGB_fused, NIR_fused, SWIR_fused], dim=1)
        multimodal_positions = torch.cat([self.RGB_pos, self.NIR_pos, self.SWIR_pos], dim=1)

        multimodal_inter = self.multimodal_transformer(
            torch.cat([multimodal_tokens, fused6_token], dim=1),
            torch.cat([multimodal_positions, self.fused6_pos], dim=1)
        )

        x6_inter = self.multimodal_decode_conv(
            multimodal_inter.view(B, patch_size, patch_size, patch_size, transformer_basic_dims*4)
            .permute(0,4,1,2,3)
            .contiguous()
        )   # [B, 192, 8, 8, 8]

        # -------------------------------------------------
        # 8) SHARED + PRIVATE LATENT BİRLEŞTİRME
        # -------------------------------------------------
        # Shared latent: x6_inter → 96 kanal
        z_shared = self.shared_conv(x6_inter)   # [B, 96, 8, 8, 8]

        # Private latent'ler: modality-specific inter temsillerinden
        rgb_priv  = self.rgb_private_conv(x6_RGB_)    # [B, 32, 8, 8, 8]
        nir_priv  = self.nir_private_conv(x6_NIR_)    # [B, 32, 8, 8, 8]
        swir_priv = self.swir_private_conv(x6_SWIR_)  # [B, 32, 8, 8, 8]

        # Mask varsa private'ları tekrar gate et (eksik modality → private=0)
        if mask is not None:
            rgb_mask  = mask[:, 0].view(B, 1, 1, 1, 1).to(rgb_priv.dtype).to(rgb_priv.device)
            nir_mask  = mask[:, 1].view(B, 1, 1, 1, 1).to(nir_priv.dtype).to(nir_priv.device)
            swir_mask = mask[:, 2].view(B, 1, 1, 1, 1).to(swir_priv.dtype).to(swir_priv.device)

            rgb_priv  = rgb_priv  * rgb_mask
            nir_priv  = nir_priv  * nir_mask
            swir_priv = swir_priv * swir_mask

        # Shared + private concat → toplam 192 kanal, decoder ile tam uyumlu
        x6_total = torch.cat([z_shared, rgb_priv, nir_priv, swir_priv], dim=1)
        # x6_total.shape = [B, 192, 8, 8, 8]

        # -------------------------------------------------
        # 9) İKİ DECODER + GATING
        # -------------------------------------------------
        # Her iki decoder da aynı fused_x1..x4 ve x6_total'ı görüyor;
        # gradient routing mask'e göre:
        #   - full [1,1,1] samples → full decoder
        #   - diğer tüm missing kombinasyonları → robust decoder

        pred = self.decoder_fuse(fused_x1, fused_x2, fused_x3, fused_x4, x6_total)
        return pred

