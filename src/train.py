"""
Training loop implementation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple
import os


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """Train for one epoch and return loss and accuracy."""
    
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False)
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Stats
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """Evaluate on validation set."""
    
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]', leave=False)
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    device: torch.device = None,
    save_dir: str = './checkpoints',
    early_stopping_patience: int = 5
) -> Dict:
    """
    Main training function with early stopping.
    Returns training history.
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Device: {device}")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    from .model import get_optimizer, get_scheduler
    optimizer = get_optimizer(model, learning_rate)
    scheduler = get_scheduler(optimizer, num_epochs)
    
    os.makedirs(save_dir, exist_ok=True)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 50)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {current_lr:.2e}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print(f"  New best model saved (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stopping_patience})")
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
    
    print("=" * 50)
    print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")
    
    return history
