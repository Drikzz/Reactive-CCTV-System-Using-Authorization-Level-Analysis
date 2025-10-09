"""ArcFace model implementation with ResNet backbone."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from insightface.app import FaceAnalysis


class Backbone(nn.Module):
    """ResNet backbone for feature extraction."""
    
    def __init__(self, arch='resnet50', embedding_size=512, pretrained=True, dropout=0.0):
        super().__init__()
        self.embedding_size = embedding_size
        
        # Load pretrained ResNet
        if arch == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif arch == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif arch == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif arch == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        # Remove the final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Add custom embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(feature_dim, embedding_size),
            nn.BatchNorm1d(embedding_size, eps=1e-05)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Normalized embeddings of shape (B, embedding_size)
        """
        x = self.model(x)
        x = self.embedding_layer(x)
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        return x


class ArcMarginProduct(nn.Module):
    """ArcFace: Additive Angular Margin Loss for Deep Face Recognition."""
    
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        """
        Args:
            in_features: Embedding dimension
            out_features: Number of classes
            s: Scale factor
            m: Angular margin
            easy_margin: Whether to use easy margin
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        
        # Learnable weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Pre-computed values for efficiency
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    
    def forward(self, input, label):
        """
        Forward pass.
        
        Args:
            input: Normalized embeddings (B, in_features)
            label: Ground truth labels (B,)
            
        Returns:
            Scaled logits with angular margin (B, out_features)
        """
        # Compute cosine similarity
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # Compute sine
        sine = torch.sqrt(1.0 - torch.clamp(cosine.pow(2), 0, 1))
        
        # Compute cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Convert labels to one-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        
        # Apply margin only to the correct class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale the output
        output *= self.s
        
        return output


class ArcFaceModel(nn.Module):
    """Complete ArcFace model combining backbone and margin head."""
    
    def __init__(self, num_classes, arch='resnet50', embedding_size=512, 
                 s=30.0, m=0.50, easy_margin=False, pretrained=True, dropout=0.0):
        """
        Args:
            num_classes: Number of identities in training set
            arch: Backbone architecture
            embedding_size: Dimension of face embeddings
            s: Scale factor for ArcMargin
            m: Angular margin for ArcMargin
            easy_margin: Whether to use easy margin
            pretrained: Whether to use pretrained backbone
            dropout: Dropout rate
        """
        super().__init__()
        
        self.backbone = Backbone(
            arch=arch,
            embedding_size=embedding_size,
            pretrained=pretrained,
            dropout=dropout
        )
        
        self.margin = ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            s=s,
            m=m,
            easy_margin=easy_margin
        )
        
        self.embedding_size = embedding_size
        self.num_classes = num_classes
    
    def forward(self, x, labels=None):
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, H, W)
            labels: Ground truth labels (B,) - required for training
            
        Returns:
            If labels provided: margin logits for training
            If no labels: normalized embeddings for inference
        """
        embeddings = self.backbone(x)
        
        if labels is not None:
            # Training mode - return margin logits
            return self.margin(embeddings, labels)
        else:
            # Inference mode - return embeddings
            return embeddings
    
    def get_embeddings(self, x):
        """Get embeddings without margin (for inference)."""
        return self.backbone(x)


class ImprovedArcFaceSystem:
    def __init__(self):
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.known_faces = {}  # Store known face embeddings
    
    def recognize_face(self, image):
        faces = self.app.get(image)
        if not faces:
            return 'Unknown', 0.0
        
        # Get the most confident face
        face = max(faces, key=lambda x: x.det_score)
        embedding = face.normed_embedding
        
        # Compare with known faces
        best_match = 'Unknown'
        best_similarity = 0.0
        
        for name, known_embedding in self.known_faces.items():
            similarity = np.dot(embedding, known_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        if best_similarity > 0.6:  # Adjust threshold
            return best_match, best_similarity
        else:
            return 'Unknown', best_similarity


# Utility functions
def l2_norm(input, axis=1):
    """L2 normalization."""
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def cosine_similarity(x1, x2):
    """Compute cosine similarity between two embeddings."""
    return F.cosine_similarity(x1, x2, dim=-1)


def euclidean_distance(x1, x2):
    """Compute Euclidean distance between two embeddings."""
    return torch.norm(x1 - x2, p=2, dim=-1)


def verify_embeddings(emb1, emb2, metric='cosine'):
    """
    Verify if two embeddings belong to the same person.
    
    Args:
        emb1, emb2: Normalized embeddings
        metric: 'cosine' or 'euclidean'
        
    Returns:
        Similarity/distance score
    """
    if metric == 'cosine':
        return cosine_similarity(emb1, emb2)
    elif metric == 'euclidean':
        return euclidean_distance(emb1, emb2)
    else:
        raise ValueError(f"Unknown metric: {metric}")