import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if  model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
 
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
    
    #PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, weight=0.5):
        
        #comment out if  model contains a sigmoid or equivalent activation layer
        inputs_sigmoid = F.sigmoid(inputs)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        inputs_sigmoid = inputs_sigmoid.view(-1)
        targets = targets.view(-1)
        
        # Debug prints
        print("Inputs_sigmoid range:", inputs_sigmoid.min().item(), inputs_sigmoid.max().item())
        print("Targets sum:", targets.sum().item())
        print("Inputs_sigmoid sum:", inputs_sigmoid.sum().item())
        
        intersection = (inputs_sigmoid * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_sigmoid.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')

        print("Intersection:", intersection.item())
        print("Dice Loss:", dice_loss.item())
        print("BCE Loss:", BCE.item())
        print("Weight:", weight)

        Dice_BCE = weight*BCE + (1-weight)*dice_loss
        
        print("Total Dice_BCE Loss:", Dice_BCE.item())
        
        return Dice_BCE
    
#PyTorch
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
    
#PyTorch
ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):      
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)  
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
    

