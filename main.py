#!/usr/bin/env python3
"""
Image Classifier with Transfer Learning

Fine-tunes pre-trained ResNet/EfficientNet models on Flowers102 dataset.

Usage:
    python main.py --model resnet50 --epochs 20
    python main.py --help
"""

import argparse
import torch
import os
import json
from datetime import datetime

from src.dataset import load_flowers102
from src.model import create_model
from src.train import train
from src.evaluate import generate_report


def parse_args():
    parser = argparse.ArgumentParser(
        description='Image Classifier with Transfer Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet18', 'resnet50', 'efficientnet_b0'],
                       help='Model architecture')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone weights')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Results directory')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate existing model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path for evaluation')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 50)
    print("IMAGE CLASSIFIER WITH TRANSFER LEARNING")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 50 + "\n")
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load data
    train_loader, val_loader, test_loader, num_classes = load_flowers102(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    model, _ = create_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=args.freeze_backbone
    )
    
    if args.eval_only:
        checkpoint_path = args.checkpoint or os.path.join(args.save_dir, 'best_model.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            return
        
        print(f"\nLoading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        print(f"Loaded model from epoch {checkpoint['epoch']} (val_acc: {checkpoint['val_acc']:.2f}%)")
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    else:
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            device=device,
            save_dir=args.save_dir,
            early_stopping_patience=args.patience
        )
        
        print("\nLoading best model for final evaluation...")
        checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    
    # Generate evaluation report
    metrics = generate_report(
        model=model,
        test_loader=test_loader,
        history=history,
        device=device,
        results_dir=args.results_dir
    )
    
    # Save config
    config = {
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'freeze_backbone': args.freeze_backbone,
        'timestamp': datetime.now().isoformat(),
        'final_metrics': {
            'top1_accuracy': float(metrics['top1_accuracy']),
            'top5_accuracy': float(metrics['top5_accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score'])
        }
    }
    
    with open(os.path.join(args.results_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Model saved to: {args.save_dir}/best_model.pth")
    print(f"Results saved to: {args.results_dir}/")
    print("=" * 50)
    print(f"\nFinal Test Results:")
    print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
    print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
    print(f"  F1-Score: {metrics['f1_score']:.2f}%")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()
