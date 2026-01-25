"""
Model evaluation and visualization.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score
)
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import os


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get model predictions on a dataset."""
    
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in tqdm(data_loader, desc='Generating predictions'):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict:
    """Evaluate model and compute metrics."""
    
    print("Evaluating model on test set...")
    
    predictions, labels, probabilities = get_predictions(model, test_loader, device)
    
    top1_acc = (predictions == labels).mean() * 100
    top5_acc = top_k_accuracy_score(labels, probabilities, k=5) * 100
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    
    metrics = {
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'predictions': predictions,
        'labels': labels,
        'probabilities': probabilities
    }
    
    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"Precision:      {precision*100:.2f}%")
    print(f"Recall:         {recall*100:.2f}%")
    print(f"F1-Score:       {f1*100:.2f}%")
    print("=" * 40)
    
    return metrics


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: str = './results/confusion_matrix.png',
    num_classes_to_show: int = 20
):
    """Create and save confusion matrix plot."""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(labels, predictions)
    
    # For 102 classes, show only the most frequent ones
    if cm.shape[0] > num_classes_to_show:
        class_counts = np.bincount(labels)
        top_classes = np.argsort(class_counts)[-num_classes_to_show:]
        
        mask = np.isin(labels, top_classes)
        filtered_labels = labels[mask]
        filtered_preds = predictions[mask]
        
        label_map = {old: new for new, old in enumerate(sorted(top_classes))}
        filtered_labels = np.array([label_map[l] for l in filtered_labels])
        filtered_preds = np.array([label_map.get(p, -1) for p in filtered_preds])
        
        cm_display = confusion_matrix(
            filtered_labels[filtered_preds >= 0], 
            filtered_preds[filtered_preds >= 0]
        )
        title = f'Confusion Matrix (Top {num_classes_to_show} Classes)'
    else:
        cm_display = cm
        title = 'Confusion Matrix'
    
    cm_normalized = cm_display.astype('float') / cm_display.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=False, cmap='Blues', cbar_kws={'label': 'Proportion'})
    plt.title(title, fontsize=14)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved: {save_path}")


def plot_training_history(
    history: Dict,
    save_path: str = './results/training_history.png'
):
    """Plot training curves."""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training history saved: {save_path}")


def plot_sample_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    save_path: str = './results/sample_predictions.png',
    num_samples: int = 16
):
    """Visualize sample predictions."""
    
    from .dataset import denormalize
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model.eval()
    
    images, labels = next(iter(data_loader))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    with torch.no_grad():
        outputs = model(images.to(device))
        probs = torch.softmax(outputs, dim=1)
        confidences, predictions = probs.max(1)
    
    predictions = predictions.cpu()
    confidences = confidences.cpu()
    
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten()
    
    for idx in range(num_samples):
        ax = axes[idx]
        
        img = denormalize(images[idx]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        pred_class = predictions[idx].item()
        true_class = labels[idx].item()
        conf = confidences[idx].item() * 100
        
        color = 'green' if pred_class == true_class else 'red'
        ax.set_title(f'Pred: {pred_class}\nTrue: {true_class}\n({conf:.1f}%)',
                    fontsize=9, color=color)
        ax.axis('off')
    
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Sample predictions saved: {save_path}")


def generate_report(
    model: nn.Module,
    test_loader: DataLoader,
    history: Dict,
    device: torch.device,
    results_dir: str = './results'
):
    """Generate full evaluation report."""
    
    print("\n" + "=" * 50)
    print("GENERATING EVALUATION REPORT")
    print("=" * 50)
    
    os.makedirs(results_dir, exist_ok=True)
    
    metrics = evaluate_model(model, test_loader, device)
    
    if history['train_loss']:
        plot_training_history(history, f'{results_dir}/training_history.png')
    
    plot_confusion_matrix(
        metrics['labels'],
        metrics['predictions'],
        f'{results_dir}/confusion_matrix.png'
    )
    
    plot_sample_predictions(
        model, test_loader, device,
        f'{results_dir}/sample_predictions.png'
    )
    
    with open(f'{results_dir}/metrics.txt', 'w') as f:
        f.write("EVALUATION METRICS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%\n")
        f.write(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%\n")
        f.write(f"Precision:      {metrics['precision']:.2f}%\n")
        f.write(f"Recall:         {metrics['recall']:.2f}%\n")
        f.write(f"F1-Score:       {metrics['f1_score']:.2f}%\n")
    
    print(f"\nReport saved to: {results_dir}/")
    
    return metrics
