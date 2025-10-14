# Standard library
from math import sqrt

# Third-party
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
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
    def __init__(self, in_dim, out_dim, kernel, padding, stride=1, bias=True):
        super(DepthWiseConv, self).__init__()
        # Depthwise Convolution
        self.DW_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,
                                 kernel_size=kernel, stride=stride, 
                                 padding=padding, groups=in_dim, bias=bias)
        # Pointwise Convolution
        self.PW_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                                 kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.DW_conv(x)
        x = self.PW_conv(x)

        return x
        

class OverlapPatchEmbedding(nn.Module):
    def __init__(self, kernel, stride, padding, in_dim, out_dim):
        super(OverlapPatchEmbedding, self).__init__()
        self.overlap_patches = nn.Unfold(kernel_size=kernel, stride=stride, padding=padding)
        self.embedding = nn.Conv2d(in_dim*kernel**2, out_dim, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.overlap_patches(x)
        n_patches = x.shape[-1]
        divider = int(sqrt(h*w / n_patches))
        x = rearrange(x, 'b c (h w) -> b c h w', h = h//divider)
        x = self.embedding(x)

        return x

class EfficientMSA(nn.Module):
    # same size of input and output
    def __init__(self, dim, n_heads, reduction_ratio):
        super(EfficientMSA, self).__init__()
        self.reshaping_k = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.reshaping_v = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        n, c, h, w = x.shape
        LN = LayerNorm2d(c).to(device=DEVICE)
        x = LN(x)
        reshaped_k = self.reshaping_k(x)
        reshaped_v = self.reshaping_v(x)
        reshaped_k = rearrange(reshaped_k, "b c h w -> b (h w) c") # reshape (batch, sequence_length, channels) for attention
        reshaped_v = rearrange(reshaped_v, "b c h w -> b (h w) c") # reshape (batch, sequence_length, channels) for attention
        q = rearrange(x, "b c h w -> b (h w) c")
        output, output_weights = self.attention(q, reshaped_k, reshaped_v)
        output = rearrange(output, "b (h w) c -> b c h w", h=h, w=w)

        return output


class MixFFN(nn.Module):
    # same size of inputs and outputs
    def __init__(self, dim, expansion_factor):
        super(MixFFN, self).__init__()
        latent_dim = dim*expansion_factor
        self.mixffn = nn.Sequential(
            nn.Conv2d(dim, latent_dim, 1),
            DepthWiseConv(latent_dim, latent_dim, kernel=3, padding=1),
            nn.GELU(),
            nn.Conv2d(latent_dim, dim, 1)
        )
    def forward(self, x):
        n, c, h, w = x.shape
        LN = LayerNorm2d(c).to(device=DEVICE)
        x = LN(x)
        x = self.mixffn(x)
        return x
    
class MiT(nn.Module):
    def __init__(self, channels, dims, n_heads, expansion, reduction_ratio, n_layers):
        super(MiT, self).__init__()
        kernel_stride_pad = ((3, 2, 1), (3, 2, 1), (3, 2, 1), (3, 2, 1), (3, 2, 1))
        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (in_dim, out_dim), (kernel, stride, padding), n_layers, expansion, n_heads, reduction_ratio in zip(dim_pairs, kernel_stride_pad, n_layers, expansion, n_heads, reduction_ratio):
            overlapping = OverlapPatchEmbedding(kernel, stride, padding, in_dim, out_dim)
            layers = nn.ModuleList([])
            
            for _ in range(n_layers):
                layers.append(nn.ModuleList([EfficientMSA(dim=out_dim, n_heads=n_heads, reduction_ratio=reduction_ratio),
                              MixFFN(dim=out_dim, expansion_factor=expansion)]))
            self.stages.append(nn.ModuleList([overlapping, layers]))

    def forward(self, x):
        # h, w = x.shape[-2:]
        layer_outputs = []
        for overlapping, layers in self.stages:
            x = overlapping(x)  # (b, c x kernel x kernel, num_patches)
            for (attension, ffn) in layers:  # attention, feed forward
                x = attension(x) + x  # skip connection
                x = ffn(x) + x

            layer_outputs.append(x)  # multi scale features

        return layer_outputs
    


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
    def __init__(self, channels=3, dims=(64, 128, 256, 512), n_heads=(1, 2, 8, 8),  # dims được điều chỉnh
                 expansion=(8, 8, 4, 4), reduction_ratio=(8, 4, 2, 1), n_layers=(2, 2, 2, 2), 
                 learning_rate=config.LEARNING_RATE):
        super(HybridSegmentor, self).__init__()
        
        # Keep encoders
        self.mix_transformer = MiT(channels, dims, n_heads, expansion, reduction_ratio, n_layers)
        self.cnn_encoder = ResNetEncoder()

        # Điều chỉnh các BiFusion blocks
        self.fusion1 = BiFusion_block(ch_1=64, ch_2=64, r_2=4, ch_int=32, ch_out=32)        
        self.fusion2 = BiFusion_block(ch_1=64, ch_2=128, r_2=4, ch_int=64, ch_out=64)    
        self.fusion3 = BiFusion_block(ch_1=128, ch_2=256, r_2=4, ch_int=128, ch_out=128)    
        self.fusion4 = BiFusion_block(ch_1=256, ch_2=512, r_2=4, ch_int=256, ch_out=256)

        # Điều chỉnh decoder path
        self.up5 = Up(512, 256, 256, attn=True)  
        self.up4 = Up(256, 128, 128, attn=True) 
        self.up3 = Up(128, 64, 64, attn=True)
        self.up2 = Up(64, 32, 32, attn=True)   
        self.up1 = Up(32, 16, attn=False)

        # Điều chỉnh final convolution
        self.final = nn.Sequential(
            Conv(16, 8, 3, bn=True, relu=True),
            Conv(8, 1, 1, bn=False, relu=False)
        )

        # loss function
        # self.loss_fn = DiceBCELoss()
        # self.loss_fn.set_debug_mode(False)  # Tắt debug info trong quá trình training
        self.loss_fn = DiceLoss()
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
        mit_features = self.mix_transformer(x)
        cnn_features = self.cnn_encoder(x)
        
        # Fuse features
        fused1 = self.fusion1(cnn_features[0], mit_features[0])
        fused2 = self.fusion2(cnn_features[1], mit_features[1])
        fused3 = self.fusion3(cnn_features[2], mit_features[2])
        fused4 = self.fusion4(cnn_features[3], mit_features[3])
        
        # Simplified decoder path
        d5 = self.up5(cnn_features[4], fused4)
        d4 = self.up4(d5, fused3)
        d3 = self.up3(d4, fused2)
        d2 = self.up2(d3, fused1)
        d1 = self.up1(d2)
        
        # Upscale features to match resolution

        # Final prediction
        out = self.final(d1)
        return out, d1, d2, d3, d4, d5
        
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

        loss = self.loss_fn(pred, y, weight=0.2)
        # loss_recall = 1-self.recall(pred, y)
        # loss *= loss_recall
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        
        # if not self.training:
        #     # Calculate metrics
        #     accuracy = self.accuracy(pred, y)
        #     f1_score = self.f1_score(pred, y)
        #     precision = self.precision(pred, y)
        #     recall = self.recall(pred, y)
            
        #     print("\n=== After Processing ===")
        #     print(f"Pred min/max after sigmoid: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
        #     print(f"Pred unique values after threshold: {torch.unique(pred).tolist()}")
        #     print(f"Prediction sum: {pred.sum().item()}")
        #     print(f"Accuracy: {accuracy:.4f}")
        #     print(f"F1 Score: {f1_score:.4f}")
        #     print(f"Precision: {precision:.4f}")
        #     print(f"Recall: {recall:.4f}")
        #     print("========================\n")
        
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
