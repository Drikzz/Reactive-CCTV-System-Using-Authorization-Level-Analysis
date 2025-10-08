"""
Step 2: Train MobileNetV2 + LSTM Behavior Classifier
Loads frame sequences and trains a CNN-LSTM model to classify behaviors.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import random
from PIL import Image
import config


class VideoSequenceDataset(Dataset):
    """Dataset for loading sequences of person-tracked frames."""
    
    def __init__(self, sequence_paths, labels, transform=None):
        """
        Args:
            sequence_paths: List of (video_path, person_id) tuples
            labels: List of integer labels
            transform: Torchvision transforms for frames
        """
        self.sequence_paths = sequence_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.sequence_paths)
    
    def __getitem__(self, idx):
        video_path, person_id = self.sequence_paths[idx]
        label = self.labels[idx]
        
        # Get all frames for this person
        person_dir = video_path / f"person_{person_id}"
        frame_files = sorted(person_dir.glob("frame*.png"))
        
        # Sample frames to match sequence length
        if len(frame_files) > config.SEQUENCE_LENGTH:
            # Uniformly sample frames
            indices = np.linspace(0, len(frame_files) - 1, config.SEQUENCE_LENGTH, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        elif len(frame_files) < config.SEQUENCE_LENGTH:
            # Pad with the last frame
            while len(frame_files) < config.SEQUENCE_LENGTH:
                frame_files.append(frame_files[-1])
        
        # Load and transform frames
        frames = []
        for frame_path in frame_files:
            img = Image.open(frame_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        
        # Stack frames: [sequence_length, channels, height, width]
        frames_tensor = torch.stack(frames, dim=0)
        
        return frames_tensor, label


class MobileNetLSTM(nn.Module):
    """
    MobileNetV2 + LSTM model for video behavior classification.
    Uses pre-trained MobileNetV2 as a frame-level feature extractor,
    then feeds the sequence of features to LSTM for temporal modeling.
    """
    
    def __init__(self, num_classes, lstm_hidden_dim=256, lstm_num_layers=2, lstm_dropout=0.3):
        super(MobileNetLSTM, self).__init__()
        
        # Load pre-trained MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=True)
        
        # Remove the classifier (last layer)
        # MobileNetV2 feature extractor outputs 1280-dim features
        self.feature_extractor = nn.Sequential(*list(mobilenet.children())[:-1])
        
        # Freeze MobileNetV2 layers (optional - can fine-tune later)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Adaptive pooling to get consistent feature size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=config.MOBILENET_FEATURE_DIM,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(lstm_dropout),
            nn.Linear(lstm_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(lstm_dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch, sequence_length, channels, height, width]
        
        Returns:
            Output logits of shape [batch, num_classes]
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Reshape to process all frames at once
        x = x.view(batch_size * seq_len, c, h, w)
        
        # Extract features from each frame using MobileNetV2
        with torch.set_grad_enabled(self.training):
            features = self.feature_extractor(x)  # [batch*seq, 1280, 7, 7]
            features = self.avgpool(features)  # [batch*seq, 1280, 1, 1]
            features = features.view(batch_size * seq_len, -1)  # [batch*seq, 1280]
        
        # Reshape back to sequence
        features = features.view(batch_size, seq_len, -1)  # [batch, seq, 1280]
        
        # LSTM temporal modeling
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Take the last hidden state
        last_hidden = h_n[-1]  # [batch, hidden_dim]
        
        # Classification
        output = self.fc(last_hidden)
        
        return output


def load_dataset():
    """Load all frame sequences and create train/val/test splits."""
    
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    
    if not config.BEHAVIOR_FRAMES_DIR.exists():
        print(f"❌ Error: Frames directory not found: {config.BEHAVIOR_FRAMES_DIR}")
        print("Please run extract_frames.py first.")
        return None, None, None, None, None
    
    # Collect all sequences (video, person_id) and their labels
    all_sequences = []
    all_labels = []
    class_names = []
    
    # Scan all datasets
    for dataset_name in config.BEHAVIOR_DATASETS:
        dataset_path = config.BEHAVIOR_FRAMES_DIR / dataset_name
        if not dataset_path.exists():
            print(f"⚠️  Warning: Dataset not found: {dataset_path}")
            continue
        
        # Get all class directories
        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            if class_name not in class_names:
                class_names.append(class_name)
            
            class_idx = class_names.index(class_name)
            
            # Get all video directories
            video_dirs = [d for d in class_dir.iterdir() if d.is_dir()]
            
            for video_dir in video_dirs:
                # Get all person directories
                person_dirs = [d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith("person_")]
                
                for person_dir in person_dirs:
                    # Extract person ID
                    person_id = int(person_dir.name.split("_")[1])
                    
                    # Check if enough frames exist
                    frame_count = len(list(person_dir.glob("frame*.png")))
                    if frame_count >= config.MIN_TRACK_LENGTH:
                        all_sequences.append((video_dir, person_id))
                        all_labels.append(class_idx)
    
    # Sort class names for consistency
    class_names = sorted(class_names)
    
    # Remap labels based on sorted class names
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Update labels
    updated_labels = []
    for seq_path, _ in all_sequences:
        class_name = seq_path.parent.name
        updated_labels.append(class_to_idx[class_name])
    
    all_labels = updated_labels
    
    print(f"\n📊 Found {len(class_names)} classes:")
    for idx, name in enumerate(class_names):
        count = all_labels.count(idx)
        print(f"   {idx}: {name:25s} - {count:4d} sequences")
    
    print(f"\n✓ Total sequences: {len(all_sequences)}")
    
    # Split into train/val/test
    train_seqs, temp_seqs, train_labels, temp_labels = train_test_split(
        all_sequences, all_labels,
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        stratify=all_labels,
        random_state=42
    )
    
    val_seqs, test_seqs, val_labels, test_labels = train_test_split(
        temp_seqs, temp_labels,
        test_size=config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
        stratify=temp_labels,
        random_state=42
    )
    
    print(f"\n📦 Dataset split:")
    print(f"   Train: {len(train_seqs)} sequences")
    print(f"   Val:   {len(val_seqs)} sequences")
    print(f"   Test:  {len(test_seqs)} sequences")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    
    # Create datasets
    train_dataset = VideoSequenceDataset(train_seqs, train_labels, transform=train_transform)
    val_dataset = VideoSequenceDataset(val_seqs, val_labels, transform=val_transform)
    test_dataset = VideoSequenceDataset(test_seqs, test_labels, transform=val_transform)
    
    # Compute class weights for balanced training
    label_counts = Counter(train_labels)
    total = sum(label_counts.values())
    num_classes = len(label_counts)
    class_weights = torch.tensor(
        [total / (num_classes * label_counts[i]) for i in range(num_classes)],
        dtype=torch.float32
    )
    
    print("\n⚖️  Class weights (for loss function):")
    for idx, (name, weight) in enumerate(zip(class_names, class_weights)):
        print(f"   {idx}: {name:25s} - {weight:.3f}")
    
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
    print("MOBILENETV2 + LSTM BEHAVIOR CLASSIFIER TRAINING")
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
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create model
    num_classes = len(class_names)
    
    model = MobileNetLSTM(
        num_classes=num_classes,
        lstm_hidden_dim=config.LSTM_HIDDEN_DIM,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        lstm_dropout=config.LSTM_DROPOUT
    ).to(device)
    
    print(f"\n🧠 Model Architecture:")
    print(f"   Feature extractor: MobileNetV2 (pretrained)")
    print(f"   LSTM hidden dim: {config.LSTM_HIDDEN_DIM}")
    print(f"   LSTM layers: {config.LSTM_NUM_LAYERS}")
    print(f"   LSTM dropout: {config.LSTM_DROPOUT}")
    print(f"   Sequence length: {config.SEQUENCE_LENGTH} frames")
    print(f"   Classes: {num_classes}")
    
    # Loss and optimizer
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
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
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            model_path = config.LSTM_MODEL_DIR / "best_mobilenet_lstm.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
                'config': {
                    'num_classes': num_classes,
                    'lstm_hidden_dim': config.LSTM_HIDDEN_DIM,
                    'lstm_num_layers': config.LSTM_NUM_LAYERS,
                    'lstm_dropout': config.LSTM_DROPOUT,
                    'sequence_length': config.SEQUENCE_LENGTH
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
    
    checkpoint = torch.load(config.LSTM_MODEL_DIR / "best_mobilenet_lstm.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\n🎯 Test Accuracy: {test_acc:.2f}%")
    
    # Print confusion matrix
    evaluate_model(model, test_loader, device, class_names)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"✓ Best model saved to: {config.LSTM_MODEL_DIR / 'best_mobilenet_lstm.pth'}")
    print(f"✓ Best validation accuracy: {best_val_acc:.2f}%")
    print(f"✓ Test accuracy: {test_acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
