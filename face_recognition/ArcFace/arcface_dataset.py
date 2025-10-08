"""Dataset utilities for ArcFace training using existing faces dataset."""

import os
import random
from typing import List, Tuple, Dict, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class FaceDataset(Dataset):
    """
    Face dataset for ArcFace training using the existing faces folder.
    
    Uses: C:/Users/Alexa/.../datasets/faces/
    Expected structure:
    faces/
        person1/
            img1.jpg
            img2.jpg
            ...
        person2/
            img1.jpg
            img2.jpg
            ...
    """
    
    def __init__(self, root_dir: str = None, transform=None, img_size: int = 112):
        """
        Args:
            root_dir: Root directory containing person folders (defaults to project faces folder)
            transform: Optional transforms to apply
            img_size: Target image size (assumes square images)
        """
        # Default to your project's faces directory
        if root_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            root_dir = os.path.join(project_root, 'datasets', 'faces')
        
        self.root_dir = root_dir
        self.img_size = img_size
        
        print(f"Loading face dataset from: {self.root_dir}")
        
        # Discover classes and samples
        self.classes, self.class_to_idx, self.samples = self._make_dataset()
        self.num_classes = len(self.classes)
        
        # Set up transforms
        if transform is None:
            self.transform = self._default_transform()
        else:
            self.transform = transform
        
        print(f"Dataset loaded: {len(self.samples)} samples, {self.num_classes} identities")
    
    def _make_dataset(self) -> Tuple[List[str], Dict[str, int], List[Tuple[str, int]]]:
        """Discover classes and image samples."""
        classes = []
        class_to_idx = {}
        samples = []
        
        # Check if dataset exists
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Face dataset not found: {self.root_dir}")
        
        # Get all subdirectories (person names)
        for class_name in sorted(os.listdir(self.root_dir)):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                # Skip hidden directories
                if class_name.startswith('.'):
                    continue
                
                if class_name not in class_to_idx:
                    class_to_idx[class_name] = len(classes)
                    classes.append(class_name)
                
                class_idx = class_to_idx[class_name]
                
                # Find all image files in this class directory
                image_count = 0
                for filename in os.listdir(class_path):
                    if self._is_valid_image(filename):
                        img_path = os.path.join(class_path, filename)
                        # Verify the image can be opened
                        try:
                            with Image.open(img_path) as img:
                                img.verify()  # Check if image is valid
                            samples.append((img_path, class_idx))
                            image_count += 1
                        except Exception as e:
                            print(f"Skipping corrupted image: {img_path} - {e}")
                
                print(f"  Found {image_count} images for {class_name}")
        
        if len(classes) == 0:
            raise ValueError(f"No face classes found in {self.root_dir}")
        
        if len(samples) == 0:
            raise ValueError(f"No valid images found in {self.root_dir}")
        
        return classes, class_to_idx, samples
    
    def _is_valid_image(self, filename: str) -> bool:
        """Check if filename is a valid image."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        return any(filename.lower().endswith(ext) for ext in valid_extensions)
    
    def _default_transform(self):
        """Default data augmentation and preprocessing for face recognition."""
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),  # Faces can be flipped
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05  # Reduced for faces
            ),
            transforms.RandomRotation(degrees=10),  # Small rotation for faces
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],  # Standard for face recognition
                std=[0.5, 0.5, 0.5]
            )
        ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample."""
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random other sample
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
    
    def get_class_name(self, idx: int) -> str:
        """Get class name from class index."""
        return self.classes[idx]
    
    def get_class_samples(self, class_idx: int) -> List[str]:
        """Get all image paths for a specific class."""
        return [path for path, label in self.samples if label == class_idx]


class ValidationTransform:
    """Simple validation transforms without augmentation."""
    
    def __init__(self, img_size: int = 112):
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
    
    def __call__(self, image):
        return self.transform(image)


def create_data_loaders(
    train_dir: str = None,
    batch_size: int = 64,
    img_size: int = 112,
    num_workers: int = 4,
    pin_memory: bool = True,
    val_split: float = 0.2,
    min_samples_per_class: int = 2
) -> Tuple[DataLoader, DataLoader, int, List[str]]:
    """
    Create training and validation data loaders from your faces dataset.
    
    Args:
        train_dir: Training data directory (defaults to project faces folder)
        batch_size: Batch size
        img_size: Input image size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        val_split: Validation split ratio
        min_samples_per_class: Minimum samples per class to include
        
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of classes
        class_names: List of class names
    """
    
    # Use your existing faces dataset
    if train_dir is None:
        # Auto-detect project faces directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        train_dir = os.path.join(project_root, 'datasets', 'faces')
    
    # Load full dataset
    full_dataset = FaceDataset(
        root_dir=train_dir,
        img_size=img_size
    )
    
    # Filter classes with too few samples
    class_counts = {}
    for _, label in full_dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    valid_classes = {label: name for label, name in enumerate(full_dataset.classes) 
                    if class_counts.get(label, 0) >= min_samples_per_class}
    
    if len(valid_classes) == 0:
        raise ValueError(f"No classes have at least {min_samples_per_class} samples")
    
    # Filter samples to only include valid classes
    filtered_samples = [(path, label) for path, label in full_dataset.samples 
                       if label in valid_classes]
    
    print(f"Filtered dataset: {len(filtered_samples)} samples, {len(valid_classes)} valid classes")
    for label, name in valid_classes.items():
        count = class_counts[label]
        print(f"  {name}: {count} samples")
    
    # Create train/val split
    train_samples = []
    val_samples = []
    
    for class_label in valid_classes.keys():
        class_samples = [(path, label) for path, label in filtered_samples if label == class_label]
        
        # Split this class
        n_val = max(1, int(len(class_samples) * val_split))
        n_train = len(class_samples) - n_val
        
        # Shuffle samples for this class
        random.shuffle(class_samples)
        
        train_samples.extend(class_samples[:n_train])
        val_samples.extend(class_samples[n_train:])
    
    print(f"Split: {len(train_samples)} train, {len(val_samples)} validation")
    
    # Create datasets
    class TrainDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform
            
        def __len__(self):
            return len(self.samples)
            
        def __getitem__(self, idx):
            path, label = self.samples[idx]
            try:
                image = Image.open(path).convert('RGB')
                return self.transform(image), label
            except Exception as e:
                print(f"Error loading {path}: {e}")
                return self.__getitem__(random.randint(0, len(self.samples) - 1))
    
    # Training dataset with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Validation dataset without augmentation
    val_transform = ValidationTransform(img_size)
    
    train_dataset = TrainDataset(train_samples, train_transform)
    val_dataset = TrainDataset(val_samples, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Important for BatchNorm
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    class_names = [valid_classes[i] for i in sorted(valid_classes.keys())]
    
    return train_loader, val_loader, len(valid_classes), class_names


def analyze_faces_dataset(dataset_path: str = None):
    """Analyze your faces dataset statistics."""
    if dataset_path is None:
        # Auto-detect project faces directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dataset_path = os.path.join(project_root, 'datasets', 'faces')
    
    print(f"Analyzing faces dataset: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    try:
        dataset = FaceDataset(dataset_path)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Total samples: {len(dataset.samples)}")
        print(f"  Number of identities: {dataset.num_classes}")
        
        # Samples per class
        class_counts = {}
        for _, label in dataset.samples:
            class_name = dataset.get_class_name(label)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"  Average samples per identity: {len(dataset.samples) / dataset.num_classes:.1f}")
        print(f"  Min samples per identity: {min(class_counts.values())}")
        print(f"  Max samples per identity: {max(class_counts.values())}")
        
        # Show identities with sample counts
        print(f"\n👥 Identities (showing first 20):")
        for i, (class_name, count) in enumerate(sorted(class_counts.items())[:20]):
            print(f"  {i+1:2d}. {class_name}: {count} samples")
        
        if len(class_counts) > 20:
            print(f"  ... and {len(class_counts) - 20} more identities")
        
        # Check for classes with too few samples
        min_samples = 2
        few_samples = {name: count for name, count in class_counts.items() if count < min_samples}
        if few_samples:
            print(f"\n⚠️  Identities with < {min_samples} samples (will be excluded from training):")
            for name, count in few_samples.items():
                print(f"     {name}: {count} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        return False


# Quick test function
def test_faces_dataset():
    """Quick test of the faces dataset loading."""
    print("🧪 Testing faces dataset loading...")
    
    try:
        # Try to load the dataset
        success = analyze_faces_dataset()
        
        if success:
            print("\n✅ Dataset loading test passed!")
            
            # Try creating data loaders
            print("🔄 Testing data loaders...")
            train_loader, val_loader, num_classes, class_names = create_data_loaders(
                batch_size=4,  # Small batch for testing
                num_workers=0  # No multiprocessing for testing
            )
            
            # Test loading a batch
            for batch_idx, (images, labels) in enumerate(train_loader):
                print(f"   Batch {batch_idx + 1}: {images.shape}, labels: {labels.tolist()}")
                if batch_idx >= 2:  # Only test first few batches
                    break
            
            print(f"\n✅ Data loaders test passed!")
            print(f"   Classes: {num_classes}")
            print(f"   Sample class names: {class_names[:5]}")
            
        return success
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False


if __name__ == "__main__":
    # Run the test when script is executed directly
    test_faces_dataset()