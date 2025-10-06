"""
Step 2: Train LSTM Classifier on Pose Sequences
Loads pose sequences and trains an LSTM model to classify behaviors.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import config


class PoseSequenceDataset(Dataset):
    """Dataset for loading pose sequence tensors."""
    
    def __init__(self, data_paths, labels):
        """
        Args:
            data_paths: List of paths to .pt files
            labels: List of integer labels corresponding to each path
        """
        self.data_paths = data_paths
        self.labels = labels
        self.augmented_data = []
        self.augmented_start_idx = len(data_paths)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Check if this is an augmented sequence (stored in memory)
        if idx >= self.augmented_start_idx:
            aug_idx = idx - self.augmented_start_idx
            sequence = self.augmented_data[aug_idx]
        else:
            # Load from file
            sequence = torch.load(self.data_paths[idx], map_location='cpu')
            # Flatten keypoints: [num_frames, 17, 2] -> [num_frames, 34]
            sequence = sequence.view(sequence.size(0), -1)
        
        label = self.labels[idx]
        return sequence, label


def collate_fn(batch):
    """Custom collate function to pad sequences of variable length."""
    sequences, labels = zip(*batch)
    
    # Check if all sequences have the same length (fixed-length mode)
    seq_lengths = [len(seq) for seq in sequences]
    if len(set(seq_lengths)) == 1:
        # All sequences same length - stack directly (faster)
        sequences_batched = torch.stack(sequences, dim=0)
    else:
        # Variable length - pad sequences to the same length
        sequences_batched = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    labels = torch.tensor(labels, dtype=torch.long)
    
    return sequences_batched, labels


def augment_sequence(sequence, noise_std=0.01):
    """
    Simple augmentation: add Gaussian noise to keypoints.
    
    Args:
        sequence: torch.Tensor of shape [seq_len, input_dim]
        noise_std: standard deviation of Gaussian noise
        
    Returns:
        Augmented sequence tensor
    """
    noise = torch.randn_like(sequence) * noise_std
    return sequence + noise


class LSTMClassifier(nn.Module):
    """LSTM-based classifier for pose sequences."""
    
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the last hidden state
        last_hidden = h_n[-1]  # [batch, hidden_dim]
        
        # Apply dropout and fully connected layer
        out = self.dropout(last_hidden)
        out = self.fc(out)
        
        return out


def load_dataset():
    """Load all pose sequences and create train/val/test splits."""
    
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    
    if not config.POSE_CACHE_DIR.exists():
        print(f"❌ Error: Pose cache directory not found: {config.POSE_CACHE_DIR}")
        print("Please run extract_pose_sequences.py first.")
        return None, None, None, None
    
    # Get all class directories
    class_dirs = [d for d in config.POSE_CACHE_DIR.iterdir() if d.is_dir()]
    
    if len(class_dirs) == 0:
        print(f"❌ No class directories found in {config.POSE_CACHE_DIR}")
        return None, None, None, None
    
    # Create label mapping
    class_names = sorted([d.name for d in class_dirs])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"\n📊 Found {len(class_names)} classes:")
    for name, idx in class_to_idx.items():
        print(f"   {idx}: {name}")
    
    # Collect all data paths and labels
    all_paths = []
    all_labels = []
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        class_idx = class_to_idx[class_name]
        
        pt_files = list(class_dir.glob("*.pt"))
        print(f"\n   {class_name}: {len(pt_files)} sequences")
        
        for pt_file in pt_files:
            all_paths.append(pt_file)
            all_labels.append(class_idx)
    
    print(f"\n✓ Total sequences: {len(all_paths)}")
    
    # Split into train/val/test
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, 
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        stratify=all_labels,
        random_state=42
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
        stratify=temp_labels,
        random_state=42
    )
    
    print(f"\n📦 Dataset split:")
    print(f"   Train: {len(train_paths)} sequences")
    print(f"   Val:   {len(val_paths)} sequences")
    print(f"   Test:  {len(test_paths)} sequences")
    
    # Create datasets
    train_dataset = PoseSequenceDataset(train_paths, train_labels)
    val_dataset = PoseSequenceDataset(val_paths, val_labels)
    test_dataset = PoseSequenceDataset(test_paths, test_labels)
    
    # ===============================
    # CLASS WEIGHTING & AUGMENTATION
    # ===============================
    
    print("\n" + "=" * 60)
    print("CLASS BALANCING")
    print("=" * 60)
    
    # Count sequences per class
    label_counts = Counter(train_labels)
    print("\n📊 Sequence counts per class:")
    for idx, name in enumerate(class_names):
        count = label_counts[idx]
        print(f"   {idx}: {name:20s} - {count:3d} sequences")
    
    # Compute class weights: inverse frequency
    total = sum(label_counts.values())
    num_classes = len(label_counts)
    class_weights = torch.tensor(
        [total / (num_classes * label_counts[i]) for i in range(num_classes)],
        dtype=torch.float32
    )
    
    print("\n⚖️  Class weights (for loss function):")
    for idx, (name, weight) in enumerate(zip(class_names, class_weights)):
        print(f"   {idx}: {name:20s} - {weight:.3f}")
    
        # Data augmentation for smaller classes
    max_count = max(label_counts.values())
    augmentation_threshold = max_count * 0.8  # Augment if < 80% of largest class
    
    augmented_sequences = []
    augmented_labels = []
    
    print("\n🔄 Augmenting underrepresented classes...")
    for idx in range(len(train_dataset)):
        sequence, label = train_dataset[idx]
        
        if label_counts[label] < augmentation_threshold:
            # Number of augmentations: proportional to class size deficit
            num_augments = min(3, int((max_count - label_counts[label]) / label_counts[label]))
            
            for _ in range(num_augments):
                aug_seq = augment_sequence(sequence, noise_std=0.01)
                augmented_sequences.append(aug_seq)
                augmented_labels.append(label)
    
    # Add augmented data to training dataset
    if len(augmented_sequences) > 0:
        # Store augmented sequences in memory
        train_dataset.augmented_data = augmented_sequences
        train_dataset.augmented_start_idx = len(train_paths)
        
        # Extend labels list (no need to extend data_paths for augmented data)
        train_dataset.labels.extend(augmented_labels)
        
        print(f"✅ Added {len(augmented_sequences)} augmented sequences")
        
        # Update label counts
        new_label_counts = Counter(train_dataset.labels)
        print("\n📊 Updated sequence counts per class:")
        for idx, name in enumerate(class_names):
            original = label_counts[idx]
            updated = new_label_counts[idx]
            added = updated - original
            print(f"   {idx}: {name:20s} - {updated:3d} sequences (+{added})")
    else:
        print("ℹ️  No augmentation needed (classes are balanced)")
    
    return train_dataset, val_dataset, test_dataset, class_names, class_weights


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for sequences, labels in pbar:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def evaluate_model(model, dataloader, device, class_names):
    """Evaluate model and print confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print(cm)
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return all_preds, all_labels


def train_model():
    """Main training function."""
    
    print("=" * 60)
    print("LSTM BEHAVIOR CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Load dataset
    train_dataset, val_dataset, test_dataset, class_names, class_weights = load_dataset()
    
    if train_dataset is None:
        return
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create model
    input_dim = config.NUM_KEYPOINTS * 2  # 17 keypoints * 2 (x, y)
    num_classes = len(class_names)
    
    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        num_classes=num_classes,
        dropout=config.DROPOUT
    ).to(device)
    
    print(f"\n🧠 Model Architecture:")
    print(f"   Input dim:  {input_dim}")
    print(f"   Hidden dim: {config.HIDDEN_DIM}")
    print(f"   Num layers: {config.NUM_LAYERS}")
    print(f"   Dropout:    {config.DROPOUT}")
    print(f"   Classes:    {num_classes}")
    
    # Display sequence mode
    print(f"\n⚙️  Training Configuration:")
    if config.FIXED_SEQUENCE_LENGTH:
        print(f"   Sequence mode: Fixed-length ({config.FIXED_SEQUENCE_LENGTH} frames)")
        print(f"   Padding: None (all sequences same length)")
    else:
        print(f"   Sequence mode: Variable-length")
        print(f"   Padding: Dynamic (padded to batch max)")
    print(f"   Batch size: {config.BATCH_SIZE}")
    
    # Loss and optimizer
    # Move class weights to device and use in CrossEntropyLoss
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\n🚀 Starting training for {config.NUM_EPOCHS} epochs...")
    print("=" * 60)
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            model_path = config.LSTM_MODEL_DIR / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
                'config': {
                    'input_dim': input_dim,
                    'hidden_dim': config.HIDDEN_DIM,
                    'num_layers': config.NUM_LAYERS,
                    'num_classes': num_classes,
                    'dropout': config.DROPOUT
                }
            }, model_path)
            
            print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n⚠️  Early stopping triggered (patience: {config.EARLY_STOPPING_PATIENCE})")
                break
    
    # Load best model for final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    checkpoint = torch.load(config.LSTM_MODEL_DIR / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\n🎯 Test Accuracy: {test_acc:.2f}%")
    
    # Print confusion matrix
    evaluate_model(model, test_loader, device, class_names)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"✓ Best model saved to: {config.LSTM_MODEL_DIR / 'best_model.pth'}")
    print(f"✓ Best validation accuracy: {best_val_acc:.2f}%")
    print(f"✓ Test accuracy: {test_acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
