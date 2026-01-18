"""
Model creation with transfer learning.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple


def create_model(
    model_name: str = 'resnet50',
    num_classes: int = 102,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> Tuple[nn.Module, int]:
    """
    Create a pre-trained model and replace the classifier head.
    """
    
    print(f"Creating {model_name} model...")
    
    if model_name == 'resnet18':
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        else:
            model = models.resnet18(weights=None)
        
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, num_classes)
        )
        
    elif model_name == 'resnet50':
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            model = models.resnet50(weights=weights)
        else:
            model = models.resnet50(weights=None)
        
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        
    elif model_name == 'efficientnet_b0':
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            model = models.efficientnet_b0(weights=weights)
        else:
            model = models.efficientnet_b0(weights=None)
        
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, num_classes)
        )
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    if freeze_backbone:
        print("Freezing backbone weights")
        for name, param in model.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, num_features


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    differential_lr: bool = True
) -> torch.optim.Optimizer:
    """
    Create optimizer with optional differential learning rates.
    Backbone gets lower LR, classifier gets higher LR.
    """
    
    if differential_lr:
        backbone_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'fc' in name or 'classifier' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
        
        param_groups = [
            {'params': backbone_params, 'lr': learning_rate / 10},
            {'params': classifier_params, 'lr': learning_rate}
        ]
        
        print(f"Using differential LR: backbone={learning_rate/10:.1e}, classifier={learning_rate:.1e}")
    else:
        param_groups = model.parameters()
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    warmup_epochs: int = 2
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Cosine annealing scheduler with warmup.
    """
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            import math
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler
