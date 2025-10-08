"""ArcFace training script using existing faces dataset."""

import os
import sys
import time
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Import ArcFace components (using absolute imports)
try:
    from face_recognition.ArcFace.arcface_model import ArcFaceModel
    from face_recognition.ArcFace.arcface_dataset import create_data_loaders, analyze_faces_dataset
except ImportError as e:
    print(f"❌ Failed to import ArcFace components: {e}")
    print("Make sure all ArcFace modules are created and in the correct location.")
    sys.exit(1)


class ArcFaceTrainer:
    """ArcFace training engine."""
    
    def __init__(
        self,
        model_save_dir: str = "models/arcface",
        dataset_dir: str = None,
        config: Dict = None
    ):
        """
        Initialize ArcFace trainer.
        
        Args:
            model_save_dir: Directory to save trained models
            dataset_dir: Path to faces dataset (defaults to project faces folder)
            config: Training configuration dictionary
        """
        self.model_save_dir = model_save_dir
        self.dataset_dir = dataset_dir
        
        # Default training configuration
        self.config = {
            # Model parameters
            'arch': 'resnet50',
            'embedding_size': 512,
            'dropout': 0.1,
            's': 30.0,           # ArcFace scale factor
            'm': 0.50,           # ArcFace angular margin
            'easy_margin': False,
            
            # Training parameters
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 1e-4,
            'weight_decay': 5e-4,
            'scheduler_step': 10,
            'scheduler_gamma': 0.5,
            'warmup_epochs': 5,
            
            # Data parameters
            'img_size': 112,
            'val_split': 0.2,
            'min_samples_per_class': 3,
            'num_workers': 0,      # Changed from 4 to 0 for Windows compatibility
            'pin_memory': False,   # Changed from True to False for single-threaded
            
            # Training settings
            'save_every': 5,
            'eval_every': 1,
            'early_stopping_patience': 10,
            'mixed_precision': True,
            'gradient_clipping': 1.0,
        }
        
        # Update config with provided values
        if config:
            self.config.update(config)
        
        # Setup directories
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.logs_dir = os.path.join(self.model_save_dir, 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        self.writer = None
        
        # Training state
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.training_history = []
    
    def setup_model_and_data(self):
        """Setup model, data loaders, and training components."""
        print("📊 Analyzing dataset...")
        analyze_faces_dataset(self.dataset_dir)
        
        print("\n🔄 Creating data loaders...")
        self.train_loader, self.val_loader, self.num_classes, self.class_names = create_data_loaders(
            train_dir=self.dataset_dir,
            batch_size=self.config['batch_size'],
            img_size=self.config['img_size'],
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory'],
            val_split=self.config['val_split'],
            min_samples_per_class=self.config['min_samples_per_class']
        )
        
        print(f"\n🏗️  Building ArcFace model...")
        print(f"   Architecture: {self.config['arch']}")
        print(f"   Embedding size: {self.config['embedding_size']}")
        print(f"   Number of identities: {self.num_classes}")
        print(f"   ArcFace margin (m): {self.config['m']}")
        print(f"   ArcFace scale (s): {self.config['s']}")
        
        # Create model
        self.model = ArcFaceModel(
            num_classes=self.num_classes,
            arch=self.config['arch'],
            embedding_size=self.config['embedding_size'],
            s=self.config['s'],
            m=self.config['m'],
            easy_margin=self.config['easy_margin'],
            pretrained=True,
            dropout=self.config['dropout']
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config['scheduler_step'],
            gamma=self.config['scheduler_gamma']
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup mixed precision training
        if self.config['mixed_precision']:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=self.logs_dir)
        
        # Save config
        config_path = os.path.join(self.model_save_dir, 'config.json')
        with open(config_path, 'w') as f:
            config_to_save = self.config.copy()
            config_to_save['num_classes'] = self.num_classes
            config_to_save['class_names'] = self.class_names
            json.dump(config_to_save, f, indent=2)
        
        print(f"✅ Setup complete!")
        print(f"   Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Learning rate warmup
        if epoch < self.config['warmup_epochs']:
            lr_scale = (epoch + 1) / self.config['warmup_epochs']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config['learning_rate'] * lr_scale
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            if self.config['mixed_precision']:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images, labels)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                if self.config['gradient_clipping'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clipping']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, labels)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                if self.config['gradient_clipping'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clipping']
                    )
                
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100.0 * correct / total
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation {epoch+1}")
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                if self.config['mixed_precision']:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images, labels)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images, labels)
                    loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100.0 * correct / total
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{accuracy:.2f}%'
                })
        
        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'training_history': self.training_history
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.model_save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.model_save_dir, 'best_checkpoint.pth')
            shutil.copy2(checkpoint_path, best_path)
            print(f"💾 New best model saved! Validation accuracy: {self.best_val_acc:.2f}%")
        
        # Save periodic checkpoints
        if (epoch + 1) % self.config['save_every'] == 0:
            periodic_path = os.path.join(self.model_save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            shutil.copy2(checkpoint_path, periodic_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return
        
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"✅ Resumed from epoch {self.start_epoch}, best val acc: {self.best_val_acc:.2f}%")
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        print(f"\n🎯 Starting ArcFace training...")
        print(f"   Dataset: {self.dataset_dir or 'datasets/faces'}")
        print(f"   Model save dir: {self.model_save_dir}")
        print(f"   Epochs: {self.config['epochs']}")
        print(f"   Batch size: {self.config['batch_size']}")
        print(f"   Learning rate: {self.config['learning_rate']}")
        
        # Setup model and data
        self.setup_model_and_data()
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Training loop
        training_start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            if (epoch + 1) % self.config['eval_every'] == 0:
                val_metrics = self.validate_epoch(epoch)
            else:
                val_metrics = {'loss': 0.0, 'accuracy': 0.0}
            
            # Update scheduler
            if epoch >= self.config['warmup_epochs']:
                self.scheduler.step()
            
            # Record metrics
            epoch_time = time.time() - epoch_start_time
            epoch_data = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'lr': train_metrics['lr'],
                'epoch_time': epoch_time
            }
            self.training_history.append(epoch_data)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            if val_metrics['accuracy'] > 0:
                self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
                self.writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Learning_Rate', train_metrics['lr'], epoch)
            
            # Check if best model
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['epochs']} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            if val_metrics['accuracy'] > 0:
                print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"  LR: {train_metrics['lr']:.6f} | Time: {epoch_time:.1f}s")
            print(f"  Best Val Acc: {self.best_val_acc:.2f}% | No Improve: {self.epochs_without_improvement}")
            
            # Early stopping
            if self.epochs_without_improvement >= self.config['early_stopping_patience']:
                print(f"\n⏰ Early stopping after {self.config['early_stopping_patience']} epochs without improvement")
                break
        
        # Training completed
        total_time = time.time() - training_start_time
        print(f"\n🎉 Training completed!")
        print(f"   Total time: {total_time/3600:.2f} hours")
        print(f"   Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"   Model saved to: {self.model_save_dir}")
        
        # Save final training history
        history_path = os.path.join(self.model_save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        self.writer.close()
        
        return self.best_val_acc


def train_arcface(
    dataset_dir: str = None,
    model_save_dir: str = "models/arcface",
    config: Dict = None,
    resume_from: str = None
):
    """
    Convenient function to train ArcFace model.
    
    Args:
        dataset_dir: Path to faces dataset (defaults to project faces folder)
        model_save_dir: Directory to save trained models
        config: Training configuration dictionary
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Best validation accuracy achieved
    """
    trainer = ArcFaceTrainer(
        model_save_dir=model_save_dir,
        dataset_dir=dataset_dir,
        config=config
    )
    
    return trainer.train(resume_from=resume_from)


def main():
    """Enhanced training for better accuracy."""
    print("🚀 ArcFace Training Starting (Enhanced for Accuracy)...")
    print("=" * 50)
    
    # Enhanced config for better accuracy
    config = {
        'epochs': 50,           # More epochs for better convergence
        'batch_size': 8,        # Keep Windows-safe
        'learning_rate': 5e-4,  # Lower learning rate for stability
        'weight_decay': 1e-4,   # Better regularization
        'num_workers': 0,       # Windows-safe
        'pin_memory': False,    # Windows-safe
        'mixed_precision': False, # Disable for stability
        
        # Better training schedule
        'scheduler_step': 15,   # Reduce LR every 15 epochs
        'scheduler_gamma': 0.5, # Reduce LR by half
        'warmup_epochs': 3,     # Warm up for stability
        
        # Better validation
        'save_every': 5,
        'eval_every': 1,
        'early_stopping_patience': 15,  # More patience
        
        # ArcFace parameters for better accuracy
        's': 30.0,             # Scale parameter
        'm': 0.50,             # Margin parameter
        'embedding_size': 512,  # Feature dimension
        
        # Data augmentation settings
        'val_split': 0.15,      # Use more data for training
        'min_samples_per_class': 5,  # Ensure quality
    }
    
    print("📊 Enhanced Training Configuration:")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   ArcFace Scale (s): {config['s']}")
    print(f"   ArcFace Margin (m): {config['m']}")
    print(f"   Embedding Size: {config['embedding_size']}")
    print()
    
    # Create trainer with enhanced settings
    trainer = ArcFaceTrainer(
        model_save_dir="models/arcface",
        dataset_dir="datasets/faces",
        config=config
    )
    
    try:
        # Train with enhanced settings
        best_acc = trainer.train()
        
        print("\n" + "=" * 50)
        print("🎉 Enhanced ArcFace Training Completed!")
        print(f"   Best Validation Accuracy: {best_acc:.2f}%")
        print(f"   Model saved to: models/arcface/")
        print("=" * 50)
        
        return best_acc
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def quick_train(epochs: int = 10):
    """Quick training with even smaller settings for testing."""
    print("🚀 Quick ArcFace Training...")
    
    config = {
        'epochs': epochs,
        'batch_size': 4,        # Very small for quick testing
        'learning_rate': 1e-3,
        'num_workers': 0,       # Windows-safe
        'pin_memory': False,    # Windows-safe
        'mixed_precision': False, # Disable for stability
        'save_every': 2,
        'eval_every': 1,
        'early_stopping_patience': 5,
        'warmup_epochs': 0      # Skip warmup for quick training
    }
    
    trainer = ArcFaceTrainer(
        model_save_dir="models/arcface",
        dataset_dir="datasets/faces",
        config=config
    )
    
    return trainer.train()


if __name__ == "__main__":
    # Simple execution like FaceNet - no argument parsing!
    print("ArcFace Training Script")
    print("Usage: python arcface_train.py")
    print("   or: from arcface_train import quick_train; quick_train(5)")
    print()
    
    # Check if dataset exists
    if not os.path.exists("datasets/faces"):
        print("❌ Dataset not found at 'datasets/faces'")
        print("   Please create dataset first using capture functionality")
        print("   Example: python arcface_main.py capture --name YourName")
        exit(1)
    
    # Count dataset
    dataset_info = []
    total_images = 0
    for person in os.listdir("datasets/faces"):
        person_dir = os.path.join("datasets/faces", person)
        if os.path.isdir(person_dir):
            count = len([f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            dataset_info.append(f"   {person}: {count} images")
            total_images += count
    
    print("📊 Dataset Summary:")
    print(f"   Total images: {total_images}")
    print(f"   Identities: {len(dataset_info)}")
    for info in dataset_info:
        print(info)
    print()
    
    if total_images < 10:
        print("⚠️  Very small dataset! Consider capturing more images.")
        print("   Recommended: At least 20-50 images per person")
    
    # Just run training (like FaceNet!)
    main()