"""Check ArcFace dependencies and system status."""

import sys
import importlib
import subprocess
from typing import Dict, List, Tuple

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Check if a package is installed and get its version.
    
    Args:
        package_name: Package name for pip
        import_name: Import name (if different from package name)
        
    Returns:
        Tuple of (is_installed, version_info)
    """
    if import_name is None:
        import_name = package_name.replace('-', '_')
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown version')
        return True, version
    except ImportError:
        return False, 'Not installed'

def check_cuda_availability():
    """Check CUDA availability for PyTorch."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            return f"✅ CUDA Available - {device_count} device(s), Current: {current_device}"
        else:
            return "⚠️  CUDA Not Available - Using CPU only"
    except ImportError:
        return "❌ PyTorch not installed"

def check_opencv():
    """Check OpenCV installation and video codecs."""
    try:
        import cv2
        version = cv2.__version__
        
        # Test camera access
        cap = cv2.VideoCapture(0)
        has_camera = cap.isOpened()
        cap.release()
        
        camera_status = "✅ Camera accessible" if has_camera else "⚠️  Camera not accessible"
        
        return f"✅ OpenCV {version} - {camera_status}"
    except ImportError:
        return "❌ OpenCV not installed"

def main():
    """Check all ArcFace dependencies."""
    print("🔍 Checking ArcFace Dependencies...\n")
    
    # Core dependencies for ArcFace
    dependencies = [
        # PyTorch ecosystem
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        
        # Computer vision
        ('opencv-python', 'cv2'),
        ('facenet-pytorch', 'facenet_pytorch'),
        ('pillow', 'PIL'),
        
        # Scientific computing
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('scikit-learn', 'sklearn'),
        
        # Utilities
        ('tqdm', 'tqdm'),
        ('albumentations', 'albumentations'),
    ]
    
    print("📦 Package Status:")
    print("-" * 50)
    
    installed = []
    missing = []
    
    for package_name, import_name in dependencies:
        is_installed, version_info = check_package(package_name, import_name)
        
        if is_installed:
            status = f"✅ {package_name:<20} {version_info}"
            installed.append(package_name)
        else:
            status = f"❌ {package_name:<20} {version_info}"
            missing.append(package_name)
        
        print(status)
    
    print("\n" + "=" * 50)
    
    # Special checks
    print("\n🚀 System Status:")
    print("-" * 30)
    print(f"Python Version: {sys.version.split()[0]}")
    print(check_cuda_availability())
    print(check_opencv())
    
    # Check project structure
    print("\n📁 Project Structure:")
    print("-" * 30)
    
    project_files = [
        'face_recognition/ArcFace/arcface_model.py',
        'face_recognition/ArcFace/arcface_dataset.py',
        'face_recognition/ArcFace/arcface_train.py',
        'face_recognition/ArcFace/arcface_capture.py',
        'face_recognition/ArcFace/arcface_main.py',
        'datasets/faces/',
        'utils/common.py'
    ]
    
    import os
    for file_path in project_files:
        exists = os.path.exists(file_path)
        status = "✅" if exists else "❌"
        print(f"{status} {file_path}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"   ✅ Installed: {len(installed)} packages")
    print(f"   ❌ Missing: {len(missing)} packages")
    
    if missing:
        print(f"\n📥 To install missing packages:")
        print("   pip install " + " ".join(missing))
        
        # Check if we need CUDA PyTorch
        if 'torch' in missing:
            print("\n💡 For CUDA support:")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    # Test ArcFace imports
    print(f"\n🧪 Testing ArcFace Imports:")
    print("-" * 30)
    
    arcface_imports = [
        ('face_recognition.ArcFace.arcface_model', 'ArcFaceModel'),
        ('face_recognition.ArcFace.arcface_dataset', 'FaceDataset'),
        ('face_recognition.ArcFace.arcface_train', 'ArcFaceTrainer'),
        ('face_recognition.ArcFace.arcface_capture', 'ArcFaceCapture'),
        ('face_recognition.ArcFace.arcface_main', 'ArcFaceSystem'),
    ]
    
    for module_name, class_name in arcface_imports:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                print(f"✅ {module_name}.{class_name}")
            else:
                print(f"⚠️  {module_name} - {class_name} not found")
        except Exception as e:
            print(f"❌ {module_name} - {str(e)}")
    
    # Check dataset
    print(f"\n📊 Dataset Status:")
    print("-" * 30)
    
    faces_dir = 'datasets/faces'
    if os.path.exists(faces_dir):
        people = [d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))]
        total_images = 0
        
        for person in people[:5]:  # Show first 5
            person_dir = os.path.join(faces_dir, person)
            images = [f for f in os.listdir(person_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_images += len(images)
            print(f"   {person}: {len(images)} images")
        
        if len(people) > 5:
            print(f"   ... and {len(people) - 5} more people")
        
        print(f"   Total: {len(people)} people, ~{total_images} images")
    else:
        print(f"❌ No faces dataset found at {faces_dir}")
    
    print(f"\n🎯 Ready for ArcFace: {'✅ YES' if len(missing) == 0 else '❌ NO'}")
    
    if len(missing) == 0:
        print("   You can start using ArcFace!")
        print("   Try: python face_recognition/ArcFace/arcface_main.py test")

if __name__ == "__main__":
    main()