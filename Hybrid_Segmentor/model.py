# Standard library
from math import sqrt

# Third-party
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
from torchvision import models
from torchvision import transforms
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)
# from torchmetrics.segmentation import DiceScore
from einops import rearrange
import torchvision.transforms.functional as TF

# Local application
import config
from metric import DiceBCELoss, DiceLoss



DEVICE = config.DEVICE

# Layer Normalisation
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x
    
# Depth-wise CNN
class DepthWiseConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_dim, out_dim, kernel, padding, stride=1, bias=True):
        super().__init__()
        # Depthwise Convolution
        self.dw_conv = nn.Conv2d(
            in_channels=in_dim, 
            out_channels=in_dim,
            kernel_size=kernel, 
            stride=stride, 
            padding=padding, 
            groups=in_dim, 
            bias=bias
        )
        # Pointwise Convolution
        self.pw_conv = nn.Conv2d(
            in_channels=in_dim, 
            out_channels=out_dim,
            kernel_size=1, 
            bias=bias
        )
    
    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x
        

class OverlapPatchEmbedding(nn.Module):
    """Overlapping Patch Embedding with Layer Normalization"""
    def __init__(self, kernel, stride, padding, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Conv2d(
            in_dim, 
            out_dim, 
            kernel_size=kernel, 
            stride=stride, 
            padding=padding
        )
        self.norm = LayerNorm2d(out_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class EfficientSelfAttention(nn.Module):
    """Efficient Self-Attention with Sequence Reduction"""
    def __init__(self, dim, n_heads, reduction_ratio):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        
        # Sequence reduction for K and V
        self.sr = None
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, 
                kernel_size=reduction_ratio, 
                stride=reduction_ratio
            )
            self.sr_norm = LayerNorm2d(dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=n_heads, 
            batch_first=True
        )

    def forward(self, x):
        b, c, h, w = x.shape
        
        # Layer Norm
        x_norm = self.norm(x)
        
        # Query
        q = rearrange(x_norm, "b c h w -> b (h w) c")
        
        # Key and Value with reduction
        if self.sr is not None:
            kv = self.sr(x_norm)
            kv = self.sr_norm(kv)
            kv = rearrange(kv, "b c h w -> b (h w) c")
        else:
            kv = q
        
        # Self-Attention
        attn_out, _ = self.attention(q, kv, kv)
        attn_out = rearrange(attn_out, "b (h w) c -> b c h w", h=h, w=w)
        
        return attn_out

class MixFFN(nn.Module):
    """Mix-FFN with Depthwise Convolution"""
    def __init__(self, dim, expansion_factor):
        super().__init__()
        hidden_dim = dim * expansion_factor
        
        self.norm = LayerNorm2d(dim)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.dw_conv = DepthWiseConv(
            hidden_dim, hidden_dim, 
            kernel=3, padding=1
        )
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.dw_conv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
class TransformerBlock(nn.Module):
    """Transformer Block = Efficient Attention + Mix-FFN"""
    def __init__(self, dim, n_heads, expansion, reduction_ratio):
        super().__init__()
        self.attn = EfficientSelfAttention(dim, n_heads, reduction_ratio)
        self.ffn = MixFFN(dim, expansion)
    
    def forward(self, x):
        # Attention with residual
        x = x + self.attn(x)
        # Feed-forward with residual
        x = x + self.ffn(x)
        return x

class MiT(nn.Module):
    """Mix Transformer (MiT) - SegFormer Encoder"""
    def __init__(self, in_channels, dims, n_heads, expansion, reduction_ratios, n_layers):
        super().__init__()
        
        # Stage configurations
        # (kernel_size, stride, padding) for each stage
        patch_configs = [
            (7, 4, 3),  # Stage 1: 1/4
            (3, 2, 1),  # Stage 2: 1/8
            (3, 2, 1),  # Stage 3: 1/16
            (3, 2, 1),  # Stage 4: 1/32
        ]
        
        # Build stages
        self.stages = nn.ModuleList()
        
        input_dim = in_channels
        for i, (dim, n_head, exp, ratio, n_layer) in enumerate(
            zip(dims, n_heads, expansion, reduction_ratios, n_layers)
        ):
            kernel, stride, padding = patch_configs[i]
            
            # Patch embedding
            patch_embed = OverlapPatchEmbedding(
                kernel, stride, padding, input_dim, dim
            )
            
            # Transformer blocks
            blocks = nn.ModuleList([
                TransformerBlock(dim, n_head, exp, ratio)
                for _ in range(n_layer)
            ])
            
            self.stages.append(nn.ModuleList([patch_embed, blocks]))
            input_dim = dim

    def forward(self, x):
        """
        Returns multi-scale features from all stages
        """
        outputs = []
        
        for patch_embed, blocks in self.stages:
            # Patch embedding
            x = patch_embed(x)
            
            # Transformer blocks
            for block in blocks:
                x = block(x)
            
            outputs.append(x)
        
        return outputs
    

# mobileNetv3 encoder
class MobileNetV3LargeBackbone(nn.Module):
    """
    Tạo một module "xương sống" (backbone) từ MobileNetV3-Large.
    Module này sẽ trả về 4 feature map ở các mức stride /4, /8, /16, /32.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Tải mô hình MobileNetV3-Large gốc
        if pretrained:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
        else:
            weights = None
            
        original_model = models.mobilenet_v3_large(weights=weights)
        
        # Chúng ta sẽ lấy 4 đầu ra /4, /8, /16, /32
        self.stage1 = nn.Sequential(*original_model.features[0:4])   # out: H/4
        self.stage2 = nn.Sequential(*original_model.features[4:7])   # out: H/8
        self.stage3 = nn.Sequential(*original_model.features[7:13])  # out: H/16
        self.stage4 = nn.Sequential(*original_model.features[13:17]) # out: H/32

    def forward(self, x):
        # Cho đầu vào đi qua từng giai đoạn
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        
        # Trả về 4 feature map ở các tỷ lệ khác nhau
        return c1, c2, c3, c4



# resnet_encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)

# resnet_encoder = resnet50()
class ResNetEncoder(nn.Module):
    def __init__(self, encoder = resnet_encoder):
        super(ResNetEncoder, self).__init__()
        self.encoder1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu) # 64x128x128
        self.mp = encoder.maxpool
        self.encoder2 = encoder.layer1 # 64x64x64  (thay vì 256)
        self.encoder3 = encoder.layer2 # 128x32x32 (thay vì 512)
        self.encoder4 = encoder.layer3 # 256x16x16 (thay vì 1024)
        self.encoder5 = encoder.layer4 # 512x8x8   (thay vì 2048)

    def forward(self,x):
        output1 = self.encoder1(x)
        output2 = self.mp(output1)
        output2 = self.encoder2(output2)
        output3 = self.encoder3(output2)
        output4 = self.encoder4(output3)
        output5 = self.encoder5(output4)

        return output1, output2, output3, output4, output5

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()

        # channel attention for F_g, use SE Block
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1+ch_2+ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

        
    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g*W_x)

        # spatial attention for cnn branch
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in

        # channel attetion for transformer branch
        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in
        fuse = self.residual(torch.cat([g, x, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse

class CrackAM(nn.Module):
    def __init__(self, channels, rate=1, add_maxpool=False, **_):
        super(CrackAM, self).__init__()
        self.fc = nn.Conv2d(int(channels), channels, kernel_size=1, padding=0)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        max_pool_h = torch.max(x, dim=3)[0] # (N, C, H, 1)
        max_pool_v = torch.max(x, dim=2)[0] # (N, C, 1, W)
        xtmp = torch.concat((max_pool_h, max_pool_v), dim=2)  # Shape: [batch_size, channels, width+height]
        x_se = xtmp.mean((2), keepdim=True).unsqueeze(-1)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)   

class CrackSPM(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        # Tạo tensor gồm trung bình và cực đại theo kênh
        self.compress = lambda x: torch.cat(
            (torch.mean(x, dim=1, keepdim=True),
             torch.max(x, dim=1, keepdim=True)[0]),
            dim=1
        )
        # Hai nhánh strip pooling: dọc (h) và ngang (w)
        self.conv_h = nn.Conv2d(2, 1, kernel_size=(kernel_size, 1),
                                padding=(kernel_size // 2, 0))
        self.conv_w = nn.Conv2d(2, 1, kernel_size=(1, kernel_size),
                                padding=(0, kernel_size // 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)  # [B, 2, H, W]
        h_attn = self.conv_h(x_compress)
        w_attn = self.conv_w(x_compress)
        attn = self.sigmoid(h_attn + w_attn)
        return x * attn

class BiFusion_CrackAM_CrackSPM_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        """
        Phiên bản BiFusion kết hợp với CrackAM và CrackSPM.
        CrackAM thay thế cho cơ chế channel attention (SE) ban đầu trên nhánh 'x'.
        CrackSPM thay thế cho cơ chế spatial attention (SAM) ban đầu trên nhánh 'g'.
        Args:
            ch_1 (int): Số kênh của đầu vào 'g' (từ CNN, ví dụ ResNet)
            ch_2 (int): Số kênh của đầu vào 'x' (từ Transformer, ví dụ SegFormer)
            r_2 (int): Tỷ lệ giảm kênh (reduction ratio) - không còn được CrackAM sử dụng
                       nhưng giữ lại để tương thích (hoặc có thể bỏ đi).
            ch_int (int): Số kênh trung gian cho bi-linear pooling
            ch_out (int): Số kênh đầu ra
            drop_rate (float): Tỷ lệ dropout
        """
        super(BiFusion_CrackAM_CrackSPM_block, self).__init__()

        # --- Spatial attention cho nhánh 'g' (CNN) - Giữ nguyên ---
        self.crack_spm = CrackSPM(kernel_size=7)
        
        # --- Channel attention cho nhánh 'x' (Transformer) - THAY THẾ BẰNG CRACKAM ---
        # Các dòng SE cũ đã bị xóa (self.fc1, self.fc2)
        self.crack_am = CrackAM(channels=ch_2)

        # --- Bi-linear modelling - Giữ nguyên ---
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        # --- Các thành phần khác - Giữ nguyên ---
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        # Lớp residual sẽ nhận input từ 3 nhánh đã qua attention và pooling
        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

        
    def forward(self, g, x):
        # --- Bi-linear pooling - Giữ nguyên ---
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x) # Tích chập (element-wise product)

        # --- Spatial attention cho nhánh 'g' (CNN) - Giữ nguyên ---
        g = self.crack_spm(g)

        # --- Channel attention cho nhánh 'x' (Transformer) - SỬ DỤNG CRACKAM ---
        # Phần code SE cũ đã được thay thế bằng một dòng duy nhất:
        x = self.crack_am(x) # Áp dụng CrackAM
                             # (module này đã bao gồm phép nhân x * self.gate(x_se))

        # --- Final fusion - Giữ nguyên ---
        fuse = self.residual(torch.cat([g, x, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        if in_ch2 > 0:
            self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)
        else:
            self.conv = DoubleConv(in_ch1, out_ch)

        if attn:
            # Fix: Use the correct channel dimensions for attention block
            self.attn_block = Attention_block(
                F_g=in_ch1,    # Gate signal channels (from upper layer)
                F_l=in_ch2,    # Local signal channels (from skip connection)
                F_int=min(in_ch1, in_ch2) if in_ch2 > 0 else in_ch1  # Intermediate channels
            )
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        
        if x2 is not None:
            # Handle different spatial dimensions
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                           diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
            
        return self.conv(x1)

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
                )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x)+self.identity(x))


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 
    

class HybridSegmentor(pl.LightningModule):
    def __init__(self, channels=3, dims=(32, 54, 160, 256), n_heads=(1, 2, 8, 8),  # dims được điều chỉnh
                 expansion=(8, 8, 4, 4), reduction_ratio=(8, 4, 2, 1), n_layers=(2, 2, 2, 2), 
                 learning_rate=config.LEARNING_RATE):
        super(HybridSegmentor, self).__init__()
        
        # Keep encoders
        self.mix_transformer = MiT(channels, dims, n_heads, expansion, reduction_ratio, n_layers)
        self.cnn_encoder = MobileNetV3LargeBackbone(pretrained=True)

        # Điều chỉnh các BiFusion blocks
        self.fusion1 = BiFusion_CrackAM_CrackSPM_block(ch_1=24, ch_2=32, r_2=4, ch_int=32, ch_out=64)        
        self.fusion2 = BiFusion_CrackAM_CrackSPM_block(ch_1=40, ch_2=64, r_2=4, ch_int=64, ch_out=128)    
        self.fusion3 = BiFusion_CrackAM_CrackSPM_block(ch_1=112, ch_2=160, r_2=4, ch_int=128, ch_out=256)    
        self.fusion4 = BiFusion_CrackAM_CrackSPM_block(ch_1=960, ch_2=256, r_2=4, ch_int=256, ch_out=512)

        # Điều chỉnh decoder path
        self.up4 = Up(512, 256, 256, attn=True)  # Giảm channels
        self.up3 = Up(256, 128, 128, attn=True) 
        self.up2 = Up(128, 64, 64, attn=True)
        self.up1 = Up(64, 32, 0, attn=True)   
        self.up0 = Up(32, 32, attn=False)

        # Điều chỉnh final convolution
        self.final = nn.Sequential(
            Conv(32, 8, 3, bn=True, relu=True),
            Conv(8, 1, 1, bn=False, relu=False)
        )

        # loss function
        self.loss_fn = DiceBCELoss()
        # self.loss_fn.set_debug_mode(False)  # Tắt debug info trong quá trình training
        # self.loss_fn = DiceLoss()
        # self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Confusion matrix
        self.accuracy = BinaryAccuracy()
        self.f1_score = BinaryF1Score()
        self.recall = BinaryRecall()
        self.precision = BinaryPrecision()
        
        # Overlapped area metrics (Ignore Backgrounds)
        self.jaccard_ind = BinaryJaccardIndex()
        self.dice_loss_fn = DiceLoss()  # Sử dụng DiceLoss từ metric.py để tính dice score

        # LR
        self.lr = learning_rate


    def forward(self, x):
        # Encoder paths
        mit_features = self.mix_transformer(x)  # [f1, f2, f3, f4]
        cnn_features = self.cnn_encoder(x)      # [c1, c2, c3, c4]
        
        # Fuse features from both paths using BiFusion blocks
        fused1 = self.fusion1(cnn_features[0], mit_features[0])  # 24 + 64 -> 64
        fused2 = self.fusion2(cnn_features[1], mit_features[1])  # 40 + 128 -> 128
        fused3 = self.fusion3(cnn_features[2], mit_features[2])  # 112 + 320 -> 256
        fused4 = self.fusion4(cnn_features[3], mit_features[3])  # 960 + 512 -> 512

        # Decoder path with skip connections from fusion blocks
        d4 = self.up4(fused4, fused3)  # 512 -> 256
        d3 = self.up3(d4, fused2)      # 256 -> 128
        d2 = self.up2(d3, fused1)      # 128 -> 64
        d1 = self.up1(d2)              # 64 -> 32
        d0 = self.up0(d1)              # 32 -> 32
        
        # Final convolution
        out = self.final(d0)           # 32 -> 1
        
        # Return main output and intermediate features if needed
        return out, d0, d1, d2, d3, d4
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, pred, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(pred, y)
        f1_score = self.f1_score(pred, y)
        re = self.recall(pred, y)
        precision = self.precision(pred, y)
        jaccard = self.jaccard_ind(pred, y)
        # Sử dụng DiceLoss từ metric.py để tính dice score
        # DiceLoss trả về (1 - dice), nên dice = 1 - dice_loss
        dice_loss = self.dice_loss_fn(pred, y)
        dice = 1.0 - dice_loss

        self.log_dict({'train_loss': loss, 
                       'train_accuracy': accuracy, 
                       'train_f1_score': f1_score, 
                       'train_precision': precision,  
                       'train_recall': re, 
                       'train_IOU': jaccard,
                       'train_dice': dice},
                      on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, pred, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(pred, y)
        f1_score = self.f1_score(pred, y)
        re = self.recall(pred, y)
        precision = self.precision(pred, y)
        jaccard = self.jaccard_ind(pred, y)
        # Sử dụng DiceLoss từ metric.py để tính dice score
        # DiceLoss trả về (1 - dice), nên dice = 1 - dice_loss
        dice_loss = self.dice_loss_fn(pred, y)
        dice = 1.0 - dice_loss

        self.log_dict({'val_loss': loss,
                       'val_accuracy': accuracy,
                       'val_f1_score': f1_score, 
                       'val_precision': precision,
                       'val_recall': re,
                       'val_IOU': jaccard,
                       'val_dice': dice},
                      on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    
    def test_step(self, batch, batch_idx):
        loss, pred, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(pred, y)
        f1_score = self.f1_score(pred, y)
        re = self.recall(pred, y)
        precision = self.precision(pred, y)
        jaccard = self.jaccard_ind(pred, y)
        # Sử dụng DiceLoss từ metric.py để tính dice score
        # DiceLoss trả về (1 - dice), nên dice = 1 - dice_loss
        dice_loss = self.dice_loss_fn(pred, y)
        dice = 1.0 - dice_loss
        self.log_dict({'test_loss': loss,
                       'test_accuracy': accuracy,
                       'test_f1_score': f1_score, 
                       'test_precision': precision,
                       'test_recall': re,
                       'test_IOU': jaccard,
                       'test_dice': dice},
                      on_step=False, on_epoch=True, prog_bar=False, sync_dist=True) 
        return loss
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(config.DEVICE)
        y = y.float().unsqueeze(1).to(config.DEVICE)
        pred_lst = self.forward(x)
        pred = pred_lst[0]

        loss = self.loss_fn(pred, y, weight=0.5)
        # loss_recall = 1-self.recall(pred, y)
        # loss *= loss_recall
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

        return loss, pred, y
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(config.DEVICE)
        y = y.float().unsqueeze(1).to(config.DEVICE)
        pred = self.forward(x)
        preds = torch.sigmoid(pred)
        preds = (preds > 0.5).float()
        return preds
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        lr_schedule = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_schedule,
                "monitor": "val_loss",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }