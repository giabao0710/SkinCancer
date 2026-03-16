import os
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import timm
from tqdm import tqdm

# Constants
DATASET_DIR = './dataset/isic-2024-challenge'
TRAIN_METADATA_PATH = 'V:/NCKH/metadata/train_metadata.csv'
TEST_METADATA_PATH = 'V:/NCKH/metadata/test_metadata.csv'
IMG_DIR = os.path.join(DATASET_DIR, 'train-image', 'image')

# Model selection
MODEL_TYPE = 'hybrid'  # Options: 'efficientnet', 'vit', 'hybrid'

# Model parameters
SEED = 42
ERR = 1e-5
USE_SMOTE = False  # Toggle SMOTE on/off
SAMPLING_RATIO = 0.5  
K_NEIGHBORS = 5      
N_SPLITS = 3         
EARLY_STOPPING_ROUNDS = 4

# Column names
ID_COL = 'isic_id'
GROUP_COL = 'patient_id'
TARGET_COL = 'target'

# Image model parameters (for ViT, EfficientNet, and Hybrid)
IMG_MODEL_BATCH_SIZE = 16
IMG_MODEL_NUM_EPOCHS = 20
IMG_MODEL_LEARNING_RATE = 5e-5
IMG_MODEL_WEIGHT_DECAY = 1e-5
IMG_MODEL_EARLY_STOPPING_ROUNDS = 4

# Ensemble weights
ENSEMBLE_WEIGHTS = {
    MODEL_TYPE: 0.4,  # Use MODEL_TYPE instead of 'efficientnet'
    'lgb': 0.3,        
    'xgb': 0.3          
}



# EfficientNet model definition
class EfficientNetModel(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super(EfficientNetModel, self).__init__()
        self.model = timm.create_model('tf_efficientnet_b0', pretrained=pretrained)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-4]:
            param.requires_grad = False
            
        # Modify the final layers
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Add attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Get features from EfficientNet
        features = self.model.forward_features(x)
        features = self.model.global_pool(features)
        features = features.flatten(1)
        
        # Apply attention
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        # Final classification
        output = self.model.classifier(features)
        return output

# Vision Transformer model definition
class VisionTransformerModel(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super(VisionTransformerModel, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-4]:
            param.requires_grad = False
            
        # Get feature dimension from ViT
        num_features = self.model.head.in_features
        
        # Add global pooling to handle sequence length
        self.global_pool = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, num_features),
            nn.GELU()
        )
        
        # Modify the final layers
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Add attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Get features from ViT
        features = self.model.forward_features(x)  # Shape: [B, 197, 768]
        
        # Global pooling to handle sequence length
        # Use CLS token (first token) and global pooling
        cls_token = features[:, 0]  # Shape: [B, 768]
        patch_tokens = features[:, 1:]  # Shape: [B, 196, 768]
        
        # Global average pooling on patch tokens
        patch_pooled = patch_tokens.mean(dim=1)  # Shape: [B, 768]
        
        # Combine CLS token and pooled patches
        features = cls_token + patch_pooled  # Shape: [B, 768]
        features = self.global_pool(features)  # Shape: [B, 768]
        
        # Apply attention
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        # Final classification
        output = self.classifier(features)
        return output

# EfficientNet-ViT Hybrid model definition
class EfficientNetViTHybridModel(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super(EfficientNetViTHybridModel, self).__init__()
        # EfficientNet branch
        self.efficientnet = timm.create_model('tf_efficientnet_b0', pretrained=pretrained)
        for param in list(self.efficientnet.parameters())[:-4]:
            param.requires_grad = False
            
        # ViT branch
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        for param in list(self.vit.parameters())[:-4]:
            param.requires_grad = False
            
        # Feature dimensions
        eff_features = self.efficientnet.classifier.in_features
        vit_features = self.vit.head.in_features
        
        # Add global pooling for ViT features
        self.vit_pool = nn.Sequential(
            nn.LayerNorm(vit_features),
            nn.Linear(vit_features, vit_features),
            nn.GELU()
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(eff_features + vit_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Attention mechanisms
        self.eff_attention = nn.Sequential(
            nn.Linear(eff_features, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        self.vit_attention = nn.Sequential(
            nn.Linear(vit_features, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Get features from EfficientNet
        eff_features = self.efficientnet.forward_features(x)
        eff_features = self.efficientnet.global_pool(eff_features)
        eff_features = eff_features.flatten(1)  # Shape: [B, eff_features]
        
        # Get features from ViT
        vit_features = self.vit.forward_features(x)  # Shape: [B, 197, vit_features]
        
        # Global pooling for ViT features
        # Use CLS token (first token) and global pooling
        cls_token = vit_features[:, 0]  # Shape: [B, vit_features]
        patch_tokens = vit_features[:, 1:]  # Shape: [B, 196, vit_features]
        
        # Global average pooling on patch tokens
        patch_pooled = patch_tokens.mean(dim=1)  # Shape: [B, vit_features]
        
        # Combine CLS token and pooled patches
        vit_features = cls_token + patch_pooled  # Shape: [B, vit_features]
        vit_features = self.vit_pool(vit_features)  # Shape: [B, vit_features]
        
        # Apply attention
        eff_attention = self.eff_attention(eff_features)
        vit_attention = self.vit_attention(vit_features)
        
        eff_features = eff_features * eff_attention
        vit_features = vit_features * vit_attention
        
        # Concatenate features (both are now 2D tensors)
        combined_features = torch.cat([eff_features, vit_features], dim=1)
        
        # Final classification
        output = self.fusion(combined_features)
        return output

def get_model(model_type, pretrained=True, num_classes=1, device='cuda'):
    """Get model based on model type"""
    if model_type == 'efficientnet':
        model = EfficientNetModel(pretrained=pretrained, num_classes=num_classes)
    elif model_type == 'vit':
        model = VisionTransformerModel(pretrained=pretrained, num_classes=num_classes)
    elif model_type == 'hybrid':
        model = EfficientNetViTHybridModel(pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)

def save_model_checkpoint(model, model_type, save_path):
    """Save model checkpoint with proper architecture information"""
    # Log model state before saving
    log_message(f"\nSaving {model_type} model checkpoint")
    log_message(f"Model state_dict keys: {list(model.state_dict().keys())[:5]}...")
    log_message(f"Total parameters: {len(model.state_dict())}")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'model_config': {
            'pretrained': False,
            'num_classes': 1
        }
    }
    
    # Add architecture-specific information
    if model_type == 'efficientnet':
        checkpoint['model_class'] = 'EfficientNetModel'
    elif model_type == 'vit':
        checkpoint['model_class'] = 'VisionTransformerModel'
    elif model_type == 'hybrid':
        checkpoint['model_class'] = 'EfficientNetViTHybridModel'
    
    torch.save(checkpoint, save_path)
    log_message(f"Model checkpoint saved to {save_path}")

def load_model_checkpoint(checkpoint_path, device='cuda'):
    """Load model checkpoint with proper architecture handling"""
    log_message(f"\nLoading model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint['model_type']
    model_config = checkpoint['model_config']
    
    log_message(f"Loading {model_type} model with config: {model_config}")
    
    # Create model with correct architecture
    model = get_model(
        model_type=model_type,
        pretrained=False,
        num_classes=model_config['num_classes'],
        device=device
    )
    
    # Log state dict information before loading
    log_message(f"Model state_dict keys before loading: {list(model.state_dict().keys())[:5]}...")
    log_message(f"Checkpoint state_dict keys: {list(checkpoint['model_state_dict'].keys())[:5]}...")
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    log_message("Model state_dict loaded successfully")
    
    return model

# Custom dataset for image loading
class MelanomaDataset(Dataset):
    def __init__(self, df, transform=None, is_train=True):
        self.df = df
        self.img_dir = IMG_DIR  
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['isic_id'] + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # If image loading fails, return a black image
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_train:
            target = self.df.iloc[idx]['target']
            return image, target
        return image

# Image transformations
def get_transforms(phase):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),  
            transforms.ColorJitter(
                brightness=0.2,  
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),  
                shear=10  
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)  
            ], p=0.3),
            transforms.RandomApply([
                transforms.RandomAdjustSharpness(sharpness_factor=2)  
            ], p=0.3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3)  
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# Add learning rate scheduler
def get_lr_scheduler(optimizer, num_steps):
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=IMG_MODEL_LEARNING_RATE,
        epochs=IMG_MODEL_NUM_EPOCHS,
        steps_per_epoch=num_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )

# Training function for image models (ViT, EfficientNet, Hybrid)
def train_image_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, early_stopping_rounds):
    best_val_auc = 0
    best_model_state = None
    no_improve_epochs = 0
    
    # Add learning rate scheduler and gradient clipping
    scheduler = get_lr_scheduler(optimizer, len(train_loader))
    max_grad_norm = 1.0  # Define max_grad_norm for gradient clipping
    
    # Log initial model state
    log_message(f"\nInitial model state_dict keys: {list(model.state_dict().keys())[:5]}...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        for images, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            images = images.to(device)
            targets = targets.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_targets, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                images = images.to(device)
                targets = targets.float().to(device)
                
                outputs = model(images).squeeze()
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_targets, val_preds)
        
        log_message(f'Epoch {epoch+1}/{num_epochs}:')
        log_message(f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}')
        log_message(f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
        log_message(f'Learning Rate: {scheduler.get_last_lr()[0]:.2e}')
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log_message(f"\nNew best model found at epoch {epoch+1}")
            log_message(f"Best model state_dict keys: {list(best_model_state.keys())[:5]}...")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            
        if no_improve_epochs >= early_stopping_rounds:
            log_message(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Log final best model state
    if best_model_state is not None:
        log_message(f"\nFinal best model state_dict shape: {len(best_model_state)} parameters")
        log_message(f"First few keys: {list(best_model_state.keys())[:5]}")
    
    # Load best model state
    model.load_state_dict(best_model_state)
    return model, best_model_state, best_val_auc

# Feature columns
NUM_COLS = [
    'age_approx',
    'clin_size_long_diam_mm',
    'tbp_lv_A',
    'tbp_lv_Aext',
    'tbp_lv_B',
    'tbp_lv_Bext',
    'tbp_lv_C',
    'tbp_lv_Cext',
    'tbp_lv_H',
    'tbp_lv_Hext',
    'tbp_lv_L',
    'tbp_lv_Lext',
    'tbp_lv_areaMM2',
    'tbp_lv_area_perim_ratio',
    'tbp_lv_color_std_mean',
    'tbp_lv_deltaA',
    'tbp_lv_deltaB',
    'tbp_lv_deltaL',
    'tbp_lv_deltaLB',
    'tbp_lv_deltaLBnorm',
    'tbp_lv_eccentricity',
    'tbp_lv_minorAxisMM',
    'tbp_lv_nevi_confidence',
    'tbp_lv_norm_border',
    'tbp_lv_norm_color',
    'tbp_lv_perimeterMM',
    'tbp_lv_radial_color_std_max',
    'tbp_lv_stdL',
    'tbp_lv_stdLExt',
    'tbp_lv_symm_2axis',
    'tbp_lv_symm_2axis_angle',
    'tbp_lv_x',
    'tbp_lv_y',
    'tbp_lv_z',
]

NEW_NUM_COLS = [
    'lesion_size_ratio',
    'lesion_shape_index',
    'hue_contrast',
    'luminance_contrast',
    'lesion_color_difference',
    'border_complexity',
    'color_uniformity',
    'position_distance_3d',
    'perimeter_to_area_ratio',
    'area_to_perimeter_ratio',
    'lesion_visibility_score',
    'symmetry_border_consistency',
    'consistency_symmetry_border',
    'color_consistency',
    'consistency_color',
    'size_age_interaction',
    'hue_color_std_interaction',
    'lesion_severity_index',
    'shape_complexity_index',
    'color_contrast_index',
    'log_lesion_area',
    'normalized_lesion_size',
    'mean_hue_difference',
    'std_dev_contrast',
    'color_shape_composite_index',
    'lesion_orientation_3d',
    'overall_color_difference',
    'symmetry_perimeter_interaction',
    'comprehensive_lesion_index',
    'color_variance_ratio',
    'border_color_interaction',
    'border_color_interaction_2',
    'size_color_contrast_ratio',
    'age_normalized_nevi_confidence',
    'age_normalized_nevi_confidence_2',
    'color_asymmetry_index',
    'volume_approximation_3d',
    'color_range',
    'shape_color_consistency',
    'border_length_ratio',
    'age_size_symmetry_index',
    'index_age_size_symmetry',
]

CAT_COLS = ['sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location', 'tbp_lv_location_simple', 'attribution']
NORM_COLS = [f'{col}_patient_norm' for col in NUM_COLS + NEW_NUM_COLS]
SPECIAL_COLS = ['count_per_patient']
FEATURE_COLS = NUM_COLS + NEW_NUM_COLS + CAT_COLS + NORM_COLS + SPECIAL_COLS

# Model parameters
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': SEED,
    'lambda_l1': 0.08758718919397321,
    'lambda_l2': 0.0039689175176025465,
    'learning_rate': 0.03231007103195577,
    'max_depth': 4,
    'num_leaves': 103,
    'colsample_bytree': 0.8329551585827726,
    'colsample_bynode': 0.4025961355653304,
    'bagging_fraction': 0.7738954452473223,
    'bagging_freq': 4,
    'min_data_in_leaf': 85,
    'scale_pos_weight': 2.7984184778875543,
    'is_unbalance': True,  # Enable class weight balancing
    'focal_loss_alpha': 0.25,  # Focal Loss alpha parameter
    'focal_loss_gamma': 2.0,   # Focal Loss gamma parameter
    # GPU settings
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'tree_learner': 'gpu',
}

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'gpu_hist',  # Use GPU accelerated algorithm
    'random_state': SEED,
    'learning_rate': 0.08501257473292347,
    'lambda': 8.879624125465703,
    'alpha': 0.6779926606782505,
    'max_depth': 6,
    'subsample': 0.6012681388711075,
    'colsample_bytree': 0.8437772277074493,
    'colsample_bylevel': 0.5476090898823716,
    'colsample_bynode': 0.9928601203635129,
    'scale_pos_weight': 3.29440313334688,
    'maximize': True,
    'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
    # GPU settings
    'gpu_id': 0,
    'predictor': 'gpu_predictor',
}

# Custom Focal Loss implementations
def focal_loss_lgb(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss for LightGBM
    """
    a, g = alpha, gamma
    p = 1 / (1 + np.exp(-y_pred))
    pt = np.where(y_true == 1, p, 1 - p)
    w = a * np.power(1 - pt, g)
    grad = w * (pt - y_true)
    hess = w * pt * (1 - pt)
    return grad, hess

def focal_loss_objective(y_true, y_pred):
    """
    Global focal loss objective function for LightGBM
    """
    return focal_loss_lgb(
        y_true, y_pred,
        alpha=0.25,  # Fixed alpha parameter
        gamma=2.0    # Fixed gamma parameter
    )

def focal_loss_xgb(preds, dtrain, alpha=0.25, gamma=2.0):
    """
    Focal Loss for XGBoost
    """
    y_true = dtrain.get_label()
    a, g = alpha, gamma
    p = 1 / (1 + np.exp(-preds))
    pt = np.where(y_true == 1, p, 1 - p)
    w = a * np.power(1 - pt, g)
    grad = w * (pt - y_true)
    hess = w * pt * (1 - pt)
    return grad, hess

class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.columns]

def calculate_pauc(y_true, y_pred_proba, max_fpr=0.2):
    """
    Calculate partial AUC (pAUC) for a specific FPR range
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (must be between 0 and 1)
        max_fpr: Maximum false positive rate (default 0.2 for 0-20% FPR range)
    Returns:
        Dictionary containing pAUC scores for both 0.1 and 0.2 FPR ranges
    """
    # Ensure predictions are probabilities
    if np.any(y_pred_proba < 0) or np.any(y_pred_proba > 1):
        y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))  # Apply sigmoid if needed
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    
    # Handle edge cases
    if len(fpr) < 2:
        return {
            'pauc_01': 0.0,
            'pauc_01_norm': 0.0,
            'pauc_02': 0.0,
            'pauc_02_norm': 0.0
        }
    
    # Calculate pAUC for 0.1 FPR
    idx_01 = np.searchsorted(fpr, 0.1)
    if idx_01 == 0:
        pauc_01 = 0.0
    else:
        pauc_01 = auc(fpr[:idx_01], tpr[:idx_01])
    pauc_01_normalized = pauc_01 / 0.1 if pauc_01 > 0 else 0.0
    
    # Calculate pAUC for 0.2 FPR
    idx_02 = np.searchsorted(fpr, max_fpr)
    if idx_02 == 0:
        pauc_02 = 0.0
    else:
        pauc_02 = auc(fpr[:idx_02], tpr[:idx_02])
    pauc_02_normalized = pauc_02 / max_fpr if pauc_02 > 0 else 0.0
    
    return {
        'pauc_01': pauc_01,
        'pauc_01_norm': pauc_01_normalized,
        'pauc_02': pauc_02,
        'pauc_02_norm': pauc_02_normalized
    }

def plot_roc_curve(y_true, y_pred, title='ROC Curve', save_path='roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Calculate pAUC for both ranges
    pauc_scores = calculate_pauc(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})\npAUC (0-10% FPR) = {pauc_scores["pauc_01_norm"]:.4f}\npAUC (0-20% FPR) = {pauc_scores["pauc_02_norm"]:.4f}')
    
    # Add pAUC regions
    plt.axvspan(0, 0.1, alpha=0.2, color='green', label='pAUC (0-10% FPR)')
    plt.axvspan(0.1, 0.2, alpha=0.2, color='yellow', label='pAUC (10-20% FPR)')
    plt.axvspan(0.2, 0.3, alpha=0.2, color='orange', label='pAUC (20-30% FPR)')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

# Global variables
LOG_FILE = None  # Will be set in main()

def log_message(message, log_file=None):
    """Log message to file with timestamp"""
    global LOG_FILE
    if log_file is None:
        log_file = LOG_FILE
    
    if log_file is None:
        print(f"Warning: No log file specified. Message: {message}")
        return
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(full_message + '\n')
        print(full_message)
    except Exception as e:
        print(f"Error writing to log file: {str(e)}")
        print(f"Message that failed to log: {full_message}")

def read_data(path):
    # Define dtypes for columns that might have mixed types
    dtypes = {
        'tbp_lv_nevi_confidence': 'float64',
        'tbp_lv_nevi_confidence_2': 'float64'
    }
    
    df = pd.read_csv(path, low_memory=False, dtype=dtypes)
    
    col_age = ['age_approx']
    df[col_age] = df[col_age].fillna(55)
    ccols = ['sex', 'anatom_site_general']
    for col in ccols:
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)
    
    return (
        pl.from_pandas(df)
        .with_columns(
            pl.col(pl.Float64).fill_nan(pl.col(pl.Float64).median()),
        )
        .with_columns(
            lesion_size_ratio = pl.col('tbp_lv_minorAxisMM') / pl.col('clin_size_long_diam_mm'),
            lesion_shape_index = pl.col('tbp_lv_areaMM2') / (pl.col('tbp_lv_perimeterMM') ** 2),
            hue_contrast = (pl.col('tbp_lv_H') - pl.col('tbp_lv_Hext')).abs(),
            luminance_contrast = (pl.col('tbp_lv_L') - pl.col('tbp_lv_Lext')).abs(),
            lesion_color_difference = (pl.col('tbp_lv_deltaA') ** 2 + pl.col('tbp_lv_deltaB') ** 2 + pl.col('tbp_lv_deltaL') ** 2).sqrt(),
            border_complexity = pl.col('tbp_lv_norm_border') + pl.col('tbp_lv_symm_2axis'),
            color_uniformity = pl.col('tbp_lv_color_std_mean') / (pl.col('tbp_lv_radial_color_std_max') + ERR),
        )
        .with_columns(
            position_distance_3d = (pl.col('tbp_lv_x') ** 2 + pl.col('tbp_lv_y') ** 2 + pl.col('tbp_lv_z') ** 2).sqrt(),
            perimeter_to_area_ratio = pl.col('tbp_lv_perimeterMM') / pl.col('tbp_lv_areaMM2'),
            area_to_perimeter_ratio = pl.col('tbp_lv_areaMM2') / pl.col('tbp_lv_perimeterMM'),
            lesion_visibility_score = pl.col('tbp_lv_deltaLBnorm') + pl.col('tbp_lv_norm_color'),
            symmetry_border_consistency = pl.col('tbp_lv_symm_2axis') * pl.col('tbp_lv_norm_border'),
            consistency_symmetry_border = pl.col('tbp_lv_symm_2axis') * pl.col('tbp_lv_norm_border') / (pl.col('tbp_lv_symm_2axis') + pl.col('tbp_lv_norm_border')),
        )
        .with_columns(
            color_consistency = pl.col('tbp_lv_stdL') / pl.col('tbp_lv_Lext'),
            consistency_color = pl.col('tbp_lv_stdL') * pl.col('tbp_lv_Lext') / (pl.col('tbp_lv_stdL') + pl.col('tbp_lv_Lext')),
            size_age_interaction = pl.col('clin_size_long_diam_mm') * pl.col('age_approx'),
            hue_color_std_interaction = pl.col('tbp_lv_H') * pl.col('tbp_lv_color_std_mean'),
            lesion_severity_index = (pl.col('tbp_lv_norm_border') + pl.col('tbp_lv_norm_color') + pl.col('tbp_lv_eccentricity')) / 3,
            shape_complexity_index = pl.col('border_complexity') + pl.col('lesion_shape_index'),
            color_contrast_index = pl.col('tbp_lv_deltaA') + pl.col('tbp_lv_deltaB') + pl.col('tbp_lv_deltaL') + pl.col('tbp_lv_deltaLBnorm'),
        )
        .with_columns(
            log_lesion_area = (pl.col('tbp_lv_areaMM2') + 1).log(),
            normalized_lesion_size = pl.col('clin_size_long_diam_mm') / pl.col('age_approx'),
            mean_hue_difference = (pl.col('tbp_lv_H') + pl.col('tbp_lv_Hext')) / 2,
            std_dev_contrast = ((pl.col('tbp_lv_deltaA') ** 2 + pl.col('tbp_lv_deltaB') ** 2 + pl.col('tbp_lv_deltaL') ** 2) / 3).sqrt(),
            color_shape_composite_index = (pl.col('tbp_lv_color_std_mean') + pl.col('tbp_lv_area_perim_ratio') + pl.col('tbp_lv_symm_2axis')) / 3,
            lesion_orientation_3d = pl.arctan2(pl.col('tbp_lv_y'), pl.col('tbp_lv_x')),
            overall_color_difference = (pl.col('tbp_lv_deltaA') + pl.col('tbp_lv_deltaB') + pl.col('tbp_lv_deltaL')) / 3,
        )
        .with_columns(
            symmetry_perimeter_interaction = pl.col('tbp_lv_symm_2axis') * pl.col('tbp_lv_perimeterMM'),
            comprehensive_lesion_index = (pl.col('tbp_lv_area_perim_ratio') + pl.col('tbp_lv_eccentricity') + pl.col('tbp_lv_norm_color') + pl.col('tbp_lv_symm_2axis')) / 4,
            color_variance_ratio = pl.col('tbp_lv_color_std_mean') / pl.col('tbp_lv_stdLExt'),
            border_color_interaction = pl.col('tbp_lv_norm_border') * pl.col('tbp_lv_norm_color'),
            border_color_interaction_2 = pl.col('tbp_lv_norm_border') * pl.col('tbp_lv_norm_color') / (pl.col('tbp_lv_norm_border') + pl.col('tbp_lv_norm_color')),
            size_color_contrast_ratio = pl.col('clin_size_long_diam_mm') / pl.col('tbp_lv_deltaLBnorm'),
            age_normalized_nevi_confidence = pl.col('tbp_lv_nevi_confidence') / pl.col('age_approx'),
            age_normalized_nevi_confidence_2 = (pl.col('clin_size_long_diam_mm')**2 + pl.col('age_approx')**2).sqrt(),
            color_asymmetry_index = pl.col('tbp_lv_radial_color_std_max') * pl.col('tbp_lv_symm_2axis'),
        )
        .with_columns(
            volume_approximation_3d = pl.col('tbp_lv_areaMM2') * (pl.col('tbp_lv_x')**2 + pl.col('tbp_lv_y')**2 + pl.col('tbp_lv_z')**2).sqrt(),
            color_range = (pl.col('tbp_lv_L') - pl.col('tbp_lv_Lext')).abs() + (pl.col('tbp_lv_A') - pl.col('tbp_lv_Aext')).abs() + (pl.col('tbp_lv_B') - pl.col('tbp_lv_Bext')).abs(),
            shape_color_consistency = pl.col('tbp_lv_eccentricity') * pl.col('tbp_lv_color_std_mean'),
            border_length_ratio = pl.col('tbp_lv_perimeterMM') / (2 * np.pi * (pl.col('tbp_lv_areaMM2') / np.pi).sqrt()),
            age_size_symmetry_index = pl.col('age_approx') * pl.col('clin_size_long_diam_mm') * pl.col('tbp_lv_symm_2axis'),
            index_age_size_symmetry = pl.col('age_approx') * pl.col('tbp_lv_areaMM2') * pl.col('tbp_lv_symm_2axis'),
        )
        .with_columns(
            ((pl.col(col) - pl.col(col).mean().over('patient_id')) / (pl.col(col).std().over('patient_id') + ERR)).alias(f'{col}_patient_norm') for col in (NUM_COLS + NEW_NUM_COLS)
        )
        .with_columns(
            count_per_patient = pl.col('isic_id').count().over('patient_id'),
        )
        .with_columns(
            pl.col(CAT_COLS).cast(pl.Categorical),
        )
        .to_pandas()
    )

def preprocess(df_train, df_test):
    global CAT_COLS
    
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32, handle_unknown='ignore')
    encoder.fit(df_train[CAT_COLS])
    
    new_cat_cols = [f'onehot_{i}' for i in range(len(encoder.get_feature_names_out()))]

    # Convert one-hot encoded features to integers
    df_train[new_cat_cols] = encoder.transform(df_train[CAT_COLS]).astype(int)
    df_test[new_cat_cols] = encoder.transform(df_test[CAT_COLS]).astype(int)

    for col in CAT_COLS:
        FEATURE_COLS.remove(col)

    FEATURE_COLS.extend(new_cat_cols)
    CAT_COLS = new_cat_cols
    
    return df_train, df_test

def custom_metric(estimator, X, y_true):
    y_hat = estimator.predict_proba(X)[:, 1]
    min_tpr = 0.80
    max_fpr = abs(1 - min_tpr)
    
    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_hat])
    
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    
    return partial_auc

def plot_fold_metrics(fold_metrics, save_dir):
    """Plot metrics comparison between folds"""
    # Plot ROC curves for each fold
    plt.figure(figsize=(10, 8))
    for fold, (fpr, tpr, roc_auc) in enumerate(fold_metrics['roc_curves'], 1):
        plt.plot(fpr, tpr, label=f'Fold {fold} (AUC = {roc_auc:.4f})', alpha=0.7)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Fold')
    plt.legend(loc="lower right")
    plt.savefig(f'{save_dir}/plots/fold_roc_curves.png')
    plt.close()
    
    # Plot pAUC histograms for both ranges
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # pAUC (0-10% FPR) histogram
    ax1.hist(fold_metrics['pauc_01_scores'], bins=5, alpha=0.7, color='green')
    ax1.axvline(np.mean(fold_metrics['pauc_01_scores']), color='r', linestyle='dashed', 
                label=f'Mean: {np.mean(fold_metrics["pauc_01_scores"]):.4f}')
    ax1.axvline(np.median(fold_metrics['pauc_01_scores']), color='g', linestyle='dashed',
                label=f'Median: {np.median(fold_metrics["pauc_01_scores"]):.4f}')
    ax1.set_xlabel('pAUC Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of pAUC Scores (0-10% FPR)')
    ax1.legend()
    
    # pAUC (0-20% FPR) histogram
    ax2.hist(fold_metrics['pauc_02_scores'], bins=5, alpha=0.7, color='blue')
    ax2.axvline(np.mean(fold_metrics['pauc_02_scores']), color='r', linestyle='dashed', 
                label=f'Mean: {np.mean(fold_metrics["pauc_02_scores"]):.4f}')
    ax2.axvline(np.median(fold_metrics['pauc_02_scores']), color='g', linestyle='dashed',
                label=f'Median: {np.median(fold_metrics["pauc_02_scores"]):.4f}')
    ax2.set_xlabel('pAUC Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of pAUC Scores (0-20% FPR)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/plots/pauc_distribution.png')
    plt.close()
    
    # Plot box plot of metrics
    metrics_df = pd.DataFrame({
        'Fold': range(1, len(fold_metrics['pauc_01_scores']) + 1),
        'pAUC (0-10% FPR)': fold_metrics['pauc_01_scores'],
        'pAUC (0-20% FPR)': fold_metrics['pauc_02_scores'],
        'ROC AUC': fold_metrics['roc_auc_scores']
    })
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=metrics_df.melt(id_vars=['Fold'], 
                                    value_vars=['pAUC (0-10% FPR)', 'pAUC (0-20% FPR)', 'ROC AUC'],
                                    var_name='Metric',
                                    value_name='Score'))
    plt.title('Distribution of Metrics Across Folds')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/plots/metrics_distribution.png')
    plt.close()

def train_with_early_stopping(X, y, groups, model_type, params, n_splits=5, early_stopping_rounds=100, eval_metric='AUC'):
    """
    Train model with k-fold cross validation and early stopping
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    models = []
    scores = []
    fold_metrics = {
        'roc_curves': [],
        'pauc_01_scores': [],
        'pauc_02_scores': [],
        'roc_auc_scores': [],
        'fold_distributions': [],
        'validation_data': []
    }
    
    # Calculate target distribution in full dataset
    full_dist = y.value_counts(normalize=True)
    log_message(f"\nFull dataset target distribution: {full_dist.to_dict()}")
    
    # Create imputer for handling NaN values
    imputer = SimpleImputer(strategy='mean')
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), 1):
        log_message(f"\nTraining fold {fold}/{n_splits}...")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        groups_train, groups_val = groups.iloc[train_idx], groups.iloc[val_idx]
        
        # Store validation data for this fold
        fold_metrics['validation_data'].append({
            'X_val': X_val,
            'y_val': y_val
        })
        
        # Handle NaN values
        log_message(f"Imputing missing values for fold {fold}...")
        X_train_imputed = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_imputed = pd.DataFrame(
            imputer.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        # Apply SMOTE only if enabled
        if USE_SMOTE:
            log_message(f"Applying SMOTE to fold {fold} training data...")
            smote = SMOTE(sampling_strategy=SAMPLING_RATIO, random_state=SEED, k_neighbors=K_NEIGHBORS)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)
            log_message(f"Training data shape after SMOTE: {X_train_resampled.shape}")
        else:
            log_message(f"SMOTE disabled, using original training data for fold {fold}")
            X_train_resampled, y_train_resampled = X_train_imputed, y_train
        
        if model_type == 'lgb':
            log_message(f"Training LightGBM model for fold {fold}...")
            model = lgb.LGBMClassifier(**params)
            model.set_params(objective=focal_loss_objective)
            
            # Create early stopping callback for LightGBM
            early_stopping_callback = lgb.early_stopping(
                stopping_rounds=early_stopping_rounds,
                verbose=False
            )
            
            model.fit(
                X_train_resampled, y_train_resampled,
                eval_set=[(X_val_imputed, y_val)],
                callbacks=[early_stopping_callback]
            )
            
            # Get raw predictions and convert to probabilities
            raw_preds = model.predict(X_val_imputed, raw_score=True)
            val_pred = 1 / (1 + np.exp(-raw_preds))  # sigmoid transformation
            
        elif model_type == 'xgb':
            log_message(f"Training XGBoost model for fold {fold}...")
            model = xgb.XGBClassifier(**params)
            
            model.fit(
                X_train_resampled, y_train_resampled,
                eval_set=[(X_val_imputed, y_val)],
                verbose=False
            )
            val_pred = model.predict_proba(X_val_imputed)[:, 1]
        
        # Calculate metrics for this fold
        roc_auc = roc_auc_score(y_val, val_pred)
        fpr, tpr, _ = roc_curve(y_val, val_pred)
        pauc_scores = calculate_pauc(y_val, val_pred)
        
        fold_metrics['roc_curves'].append((fpr, tpr, roc_auc))
        fold_metrics['pauc_01_scores'].append(pauc_scores['pauc_01_norm'])
        fold_metrics['pauc_02_scores'].append(pauc_scores['pauc_02_norm'])
        fold_metrics['roc_auc_scores'].append(roc_auc)
        
        models.append(model)
        scores.append(roc_auc)
        
        log_message(f"\nFold {fold} Results:")
        log_message(f"ROC AUC: {roc_auc:.4f}")
        log_message(f"pAUC (0-10% FPR): {pauc_scores['pauc_01']:.4f}")
        log_message(f"Normalized pAUC (0-10% FPR): {pauc_scores['pauc_01_norm']:.4f}")
        log_message(f"pAUC (0-20% FPR): {pauc_scores['pauc_02']:.4f}")
        log_message(f"Normalized pAUC (0-20% FPR): {pauc_scores['pauc_02_norm']:.4f}")
    
    return models, np.mean(scores), np.std(scores), fold_metrics

def check_gpu_availability():
    """Check GPU availability and return GPU settings"""
    gpu_available = False
    gpu_info = {}
    
    try:
        if torch.cuda.is_available():
            gpu_available = True
            gpu_info['device_name'] = torch.cuda.get_device_name(0)
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['cuda_version'] = torch.version.cuda
            log_message(f"CUDA is available. GPU Device: {gpu_info['device_name']}")
            log_message(f"Number of GPUs: {gpu_info['device_count']}")
            log_message(f"CUDA Version: {gpu_info['cuda_version']}")
        else:
            log_message("CUDA is not available. Using CPU only.")
    except ImportError:
        log_message("PyTorch not installed. Cannot check GPU availability.")
    
    # Check LightGBM GPU support
    try:
        lgb_version = lgb.__version__
        log_message(f"LightGBM version: {lgb_version}")
        if gpu_available:
            try:
                test_data = lgb.Dataset(np.random.rand(10, 5))
                test_params = {'device': 'gpu', 'tree_learner': 'gpu'}
                test_model = lgb.train(test_params, test_data, num_boost_round=1)
                gpu_info['lgb_gpu_support'] = True
                log_message("LightGBM GPU support verified")
            except Exception as e:
                gpu_info['lgb_gpu_support'] = False
                log_message(f"LightGBM GPU support not available: {str(e)}")
    except ImportError:
        log_message("LightGBM not installed properly.")
        gpu_info['lgb_gpu_support'] = False
    
    # Check XGBoost GPU support
    try:
        xgb_version = xgb.__version__
        log_message(f"XGBoost version: {xgb_version}")
        if gpu_available:
            try:
                test_data = xgb.DMatrix(np.random.rand(10, 5))
                test_params = {'tree_method': 'gpu_hist'}
                test_model = xgb.train(test_params, test_data, num_boost_round=1)
                gpu_info['xgb_gpu_support'] = True
                log_message("XGBoost GPU support verified")
            except Exception as e:
                gpu_info['xgb_gpu_support'] = False
                log_message(f"XGBoost GPU support not available: {str(e)}")
    except ImportError:
        log_message("XGBoost not installed properly.")
        gpu_info['xgb_gpu_support'] = False
    
    return gpu_available, gpu_info

def get_model_params(gpu_available, gpu_info):
    """Get model parameters based on GPU availability"""
    # Base LightGBM parameters
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': SEED,
        'lambda_l1': 0.08758718919397321,
        'lambda_l2': 0.0039689175176025465,
        'learning_rate': 0.03231007103195577,
        'max_depth': 4,
        'num_leaves': 103,
        'colsample_bytree': 0.8329551585827726,
        'colsample_bynode': 0.4025961355653304,
        'bagging_fraction': 0.7738954452473223,
        'bagging_freq': 4,
        'min_data_in_leaf': 85,
        'scale_pos_weight': 2.7984184778875543,
        'is_unbalance': True,
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
    }
    
    # Base XGBoost parameters
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': SEED,
        'learning_rate': 0.08501257473292347,
        'lambda': 8.879624125465703,
        'alpha': 0.6779926606782505,
        'max_depth': 6,
        'subsample': 0.6012681388711075,
        'colsample_bytree': 0.8437772277074493,
        'colsample_bylevel': 0.5476090898823716,
        'colsample_bynode': 0.9928601203635129,
        'scale_pos_weight': 3.29440313334688,
        'maximize': True,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
    }
    
    # Add GPU settings if available
    if gpu_available:
        if gpu_info.get('lgb_gpu_support', False):
            lgb_params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'tree_learner': 'gpu',
            })
            log_message("Using LightGBM with GPU acceleration")
        else:
            log_message("Using LightGBM with CPU (GPU not available for LightGBM)")
        
        if gpu_info.get('xgb_gpu_support', False):
            xgb_params.update({
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'predictor': 'gpu_predictor',
            })
            log_message("Using XGBoost with GPU acceleration")
        else:
            xgb_params['tree_method'] = 'hist'
            log_message("Using XGBoost with CPU (GPU not available for XGBoost)")
    else:
        xgb_params['tree_method'] = 'hist'
        log_message("Using CPU for all models")
    
    return lgb_params, xgb_params

# Add after the existing imports
def ensemble_predictions(image_preds, metadata_preds, weights=None):
    if weights is None:
        weights = ENSEMBLE_WEIGHTS
    
    # Combine predictions (removed CatBoost)
    ensemble_pred = (
        weights[MODEL_TYPE] * image_preds +
        weights['lgb'] * metadata_preds['lgb'] +
        weights['xgb'] * metadata_preds['xgb']
    )
    
    return ensemble_pred

# Add this function before main()
def get_ensemble_predictions(test_data, test_images, models_dict, device, batch_size=IMG_MODEL_BATCH_SIZE):
    """
    Get predictions from all models and combine them
    """
    # Get metadata predictions
    metadata_preds = {}
    for model_name, model in models_dict.items():
        if model_name == MODEL_TYPE:  # Skip the image model
            continue
        if model_name == 'cat':  # Skip CatBoost predictions
            continue
        if model_name == 'lgb':
            preds = model.predict(test_data, raw_score=True)
            metadata_preds[model_name] = 1 / (1 + np.exp(-preds))
        elif model_name == 'xgb':
            metadata_preds[model_name] = model.predict_proba(test_data)[:, 1]
    
    # Get image model predictions (ViT, EfficientNet, or Hybrid)
    test_dataset = MelanomaDataset(
        test_images,
        transform=get_transforms('val'),
        is_train=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    image_model = models_dict[MODEL_TYPE]
    image_model.eval()
    image_preds = []
    
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = image_model(images).squeeze()
            image_preds.extend(torch.sigmoid(outputs).cpu().numpy())
    
    image_preds = np.array(image_preds)
    
    # Combine all predictions
    ensemble_pred = ensemble_predictions(image_preds, metadata_preds)
    
    return {
        'ensemble': ensemble_pred,
        MODEL_TYPE: image_preds,
        **metadata_preds
    }

def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate various classification metrics including precision and confusion matrix
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold (default 0.5)
    Returns:
        Dictionary containing various metrics
    """
    # Convert probabilities to binary predictions using threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate precision
    precision = precision_score(y_true, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Calculate ROC and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Calculate pAUC
    pauc_scores = calculate_pauc(y_true, y_pred_proba)
    
    return {
        'precision': precision,
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'pauc_scores': pauc_scores
    }

def plot_confusion_matrix(cm, model_name, save_path):
    """
    Plot and save confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_metrics_comparison(metrics_dict, save_path):
    """
    Plot comparison of different metrics across models
    """
    models = list(metrics_dict.keys())
    metrics = ['roc_auc', 'pauc_01_norm', 'pauc_02_norm']  # Removed precision from comparison
    
    # Create DataFrame for plotting
    data = []
    for model in models:
        for metric in metrics:
            if metric in ['pauc_01_norm', 'pauc_02_norm']:
                value = metrics_dict[model]['pauc_scores'][metric]
            else:
                value = metrics_dict[model][metric]
            data.append({'Model': model, 'Metric': metric, 'Value': value})
    
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Model', y='Value', hue='Metric')
    plt.title('Model Metrics Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    try:
        global LOG_FILE
        start_time = datetime.now()
        
        # Create output directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'out/dacs{timestamp}_{MODEL_TYPE}_ensemble_img_metadata'
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/text', exist_ok=True)
        os.makedirs(f'{output_dir}/plots', exist_ok=True)
        os.makedirs(f'{output_dir}/models', exist_ok=True)
        os.makedirs(f'{output_dir}/lgb/plots', exist_ok=True)
        os.makedirs(f'{output_dir}/xgb/plots', exist_ok=True)
        os.makedirs(f'{output_dir}/{MODEL_TYPE}/plots', exist_ok=True)
        
        # Set up log file
        LOG_FILE = f'{output_dir}/text/pipeline_log.txt'
        
        # Clear log file and write header
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"=== Pipeline Log Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        
        log_message("=== Starting Pipeline ===")
        log_message(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_message(f"Log file: {LOG_FILE}")
        
        # Check GPU availability and get model parameters
        log_message("\n=== Checking GPU Availability ===")
        gpu_available, gpu_info = check_gpu_availability()
        lgb_params, xgb_params = get_model_params(gpu_available, gpu_info)
        
        # Read and preprocess data
        log_message("\n=== Data Loading and Preprocessing ===")
        log_message("Reading training data...")
        df_train = read_data(TRAIN_METADATA_PATH)
        log_message(f"Training data loaded: {len(df_train)} samples")
        
        log_message("Reading test data...")
        df_test = read_data(TEST_METADATA_PATH)
        log_message(f"Test data loaded: {len(df_test)} samples")
        
        # Validate data
        log_message("\nValidating data...")
        if df_train.empty or df_test.empty:
            raise ValueError("Training or test data is empty")
        if TARGET_COL not in df_train.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found in training data")
        if GROUP_COL not in df_train.columns:
            raise ValueError(f"Group column '{GROUP_COL}' not found in training data")
        log_message("Data validation passed")
        
        # Log class distribution
        class_dist_before = df_train[TARGET_COL].value_counts(normalize=True)
        log_message("\nClass distribution before sampling:")
        for class_label, percentage in class_dist_before.items():
            log_message(f"Class {class_label}: {percentage:.4%}")
        
        # Ensure target is binary
        if df_train[TARGET_COL].nunique() > 2:
            log_message("\nWarning: Target has more than 2 unique values. Converting to binary...")
            df_train[TARGET_COL] = (df_train[TARGET_COL] > 0).astype(int)
            log_message("Target converted to binary")
        
        # Preprocess data
        log_message("\nPreprocessing data...")
        df_train, df_test = preprocess(df_train, df_test)
        log_message("Data preprocessing completed")
        
        # Ensure all feature columns are numeric
        log_message("\nConverting features to numeric...")
        for col in FEATURE_COLS:
            if df_train[col].dtype == 'object':
                df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
                df_train[col] = df_train[col].fillna(df_train[col].mean())
            if df_test[col].dtype == 'object':
                df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
                df_test[col] = df_test[col].fillna(df_test[col].mean())
        log_message("Feature conversion completed")
        
        # Prepare data
        log_message("\nPreparing data for training...")
        X_train = df_train[FEATURE_COLS]
        y_train = df_train[TARGET_COL]
        groups_train = df_train[GROUP_COL]
        
        X_test = df_test[FEATURE_COLS]
        y_test = df_test[TARGET_COL] if TARGET_COL in df_test.columns else None
        
        log_message(f"Training data prepared: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        if y_test is not None:
            log_message(f"Test data prepared: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        # Train and evaluate individual models
        log_message("\n=== Model Training and Evaluation ===")
        
        # LightGBM
        log_message("\n--- LightGBM Training ---")
        lgb_start_time = datetime.now()
        log_message(f"Starting LightGBM training at {lgb_start_time.strftime('%H:%M:%S')}")
        
        lgb_models, lgb_mean_score, lgb_std_score, lgb_fold_metrics = train_with_early_stopping(
            X_train, y_train, groups_train, 'lgb', lgb_params, n_splits=N_SPLITS, early_stopping_rounds=EARLY_STOPPING_ROUNDS
        )
        
        lgb_end_time = datetime.now()
        lgb_duration = lgb_end_time - lgb_start_time
        log_message(f"LightGBM training completed in {lgb_duration}")
        
        # Save LightGBM results
        log_message("\nSaving LightGBM results...")
        with open(f'{output_dir}/text/lgb_results.txt', 'w', encoding='utf-8') as f:
            f.write("LightGBM Model Results:\n\n")
            f.write(f"Training Duration: {lgb_duration}\n")
            f.write(f"Mean ROC AUC: {lgb_mean_score:.4f} (+/- {lgb_std_score * 2:.4f})\n")
            f.write(f"Mean pAUC (0-10% FPR): {np.mean(lgb_fold_metrics['pauc_01_scores']):.4f} (+/- {np.std(lgb_fold_metrics['pauc_01_scores']) * 2:.4f})\n")
            f.write(f"Mean pAUC (0-20% FPR): {np.mean(lgb_fold_metrics['pauc_02_scores']):.4f} (+/- {np.std(lgb_fold_metrics['pauc_02_scores']) * 2:.4f})\n")
            f.write(f"Median pAUC (0-10% FPR): {np.median(lgb_fold_metrics['pauc_01_scores']):.4f}\n")
            f.write(f"Median pAUC (0-20% FPR): {np.median(lgb_fold_metrics['pauc_02_scores']):.4f}\n\n")
            
            f.write("Fold-wise Results:\n")
            for fold, (pauc_01, pauc_02, roc_auc) in enumerate(zip(
                lgb_fold_metrics['pauc_01_scores'], 
                lgb_fold_metrics['pauc_02_scores'],
                lgb_fold_metrics['roc_auc_scores']), 1):
                f.write(f"Fold {fold} - ROC AUC: {roc_auc:.4f}, pAUC (0-10%): {pauc_01:.4f}, pAUC (0-20%): {pauc_02:.4f}\n")
        
        # Plot LightGBM metrics
        log_message("Generating LightGBM plots...")
        plot_fold_metrics(lgb_fold_metrics, f'{output_dir}/lgb')
        
        # Save best LightGBM model
        log_message("Saving best LightGBM model...")
        best_lgb_idx = np.argmax(lgb_fold_metrics['roc_auc_scores'])
        best_lgb_model = lgb_models[best_lgb_idx]
        joblib.dump(best_lgb_model, f'{output_dir}/models/best_lgb_model.joblib')
        
        # XGBoost
        log_message("\n--- XGBoost Training ---")
        xgb_start_time = datetime.now()
        log_message(f"Starting XGBoost training at {xgb_start_time.strftime('%H:%M:%S')}")
        
        xgb_models, xgb_mean_score, xgb_std_score, xgb_fold_metrics = train_with_early_stopping(
            X_train, y_train, groups_train, 'xgb', xgb_params, n_splits=N_SPLITS, early_stopping_rounds=EARLY_STOPPING_ROUNDS
        )
        
        xgb_end_time = datetime.now()
        xgb_duration = xgb_end_time - xgb_start_time
        log_message(f"XGBoost training completed in {xgb_duration}")
        
        # Save XGBoost results
        log_message("\nSaving XGBoost results...")
        with open(f'{output_dir}/text/xgb_results.txt', 'w', encoding='utf-8') as f:
            f.write("XGBoost Model Results:\n\n")
            f.write(f"Training Duration: {xgb_duration}\n")
            f.write(f"Mean ROC AUC: {xgb_mean_score:.4f} (+/- {xgb_std_score * 2:.4f})\n")
            f.write(f"Mean pAUC (0-10% FPR): {np.mean(xgb_fold_metrics['pauc_01_scores']):.4f} (+/- {np.std(xgb_fold_metrics['pauc_01_scores']) * 2:.4f})\n")
            f.write(f"Mean pAUC (0-20% FPR): {np.mean(xgb_fold_metrics['pauc_02_scores']):.4f} (+/- {np.std(xgb_fold_metrics['pauc_02_scores']) * 2:.4f})\n")
            f.write(f"Median pAUC (0-10% FPR): {np.median(xgb_fold_metrics['pauc_01_scores']):.4f}\n")
            f.write(f"Median pAUC (0-20% FPR): {np.median(xgb_fold_metrics['pauc_02_scores']):.4f}\n\n")
            
            f.write("Fold-wise Results:\n")
            for fold, (pauc_01, pauc_02, roc_auc) in enumerate(zip(
                xgb_fold_metrics['pauc_01_scores'], 
                xgb_fold_metrics['pauc_02_scores'],
                xgb_fold_metrics['roc_auc_scores']), 1):
                f.write(f"Fold {fold} - ROC AUC: {roc_auc:.4f}, pAUC (0-10%): {pauc_01:.4f}, pAUC (0-20%): {pauc_02:.4f}\n")
        
        # Plot XGBoost metrics
        log_message("Generating XGBoost plots...")
        plot_fold_metrics(xgb_fold_metrics, f'{output_dir}/xgb')
        
        # Save best XGBoost model
        log_message("Saving best XGBoost model...")
        best_xgb_idx = np.argmax(xgb_fold_metrics['roc_auc_scores'])
        best_xgb_model = xgb_models[best_xgb_idx]
        joblib.dump(best_xgb_model, f'{output_dir}/models/best_xgb_model.joblib')
        
        # Initialize model based on model type
        log_message(f"\n=== {MODEL_TYPE.upper()} Training and Evaluation ===")
        model_start_time = datetime.now()
        log_message(f"Starting {MODEL_TYPE} training at {model_start_time.strftime('%H:%M:%S')}")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_message(f"Using device: {device}")
        
        # Create cross-validation splits
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        model_models = []
        model_fold_metrics = {
            'roc_curves': [],
            'pauc_scores': [],
            'pauc_01_scores': [],
            'pauc_02_scores': [],
            'roc_auc_scores': [],
            'validation_data': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups_train), 1):
            log_message(f"\nTraining {MODEL_TYPE} fold {fold}/{N_SPLITS}...")
            
            # Create new model instance for each fold
            model = get_model(MODEL_TYPE, pretrained=True, device=device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.AdamW(model.parameters(), lr=IMG_MODEL_LEARNING_RATE, weight_decay=IMG_MODEL_WEIGHT_DECAY)
            
            # Create datasets
            train_dataset = MelanomaDataset(
                df_train.iloc[train_idx],
                transform=get_transforms('train'),
                is_train=True
            )
            val_dataset = MelanomaDataset(
                df_train.iloc[val_idx],
                transform=get_transforms('val'),
                is_train=True
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=IMG_MODEL_BATCH_SIZE,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=IMG_MODEL_BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # Train model
            model, best_model_state, val_auc = train_image_model(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                device,
                IMG_MODEL_NUM_EPOCHS,
                IMG_MODEL_EARLY_STOPPING_ROUNDS
            )
            
            # Store model state and validation data
            model_models.append(best_model_state)
            model_fold_metrics['validation_data'].append({
                'X_val': df_train.iloc[val_idx],
                'y_val': y_train.iloc[val_idx]
            })
            
            # Calculate metrics
            model.eval()
            val_preds = []
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(device)
                    outputs = model(images).squeeze()
                    val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            
            val_preds = np.array(val_preds)
            val_true = y_train.iloc[val_idx].values
            
            # Calculate metrics
            metrics = calculate_metrics(val_true, val_preds)
            
            model_fold_metrics['roc_curves'].append((metrics['fpr'], metrics['tpr'], metrics['roc_auc']))
            model_fold_metrics['pauc_scores'].append(metrics['pauc_scores'])
            model_fold_metrics['pauc_01_scores'].append(metrics['pauc_scores']['pauc_01_norm'])
            model_fold_metrics['pauc_02_scores'].append(metrics['pauc_scores']['pauc_02_norm'])
            model_fold_metrics['roc_auc_scores'].append(metrics['roc_auc'])
            
            log_message(f"\n{MODEL_TYPE.upper()} Fold {fold} Results:")
            log_message(f"ROC AUC: {metrics['roc_auc']:.4f}")
            log_message(f"pAUC (0-10% FPR): {metrics['pauc_scores']['pauc_01']:.4f}")
            log_message(f"Normalized pAUC (0-10% FPR): {metrics['pauc_scores']['pauc_01_norm']:.4f}")
            log_message(f"pAUC (0-20% FPR): {metrics['pauc_scores']['pauc_02']:.4f}")
            log_message(f"Normalized pAUC (0-20% FPR): {metrics['pauc_scores']['pauc_02_norm']:.4f}")
        
        # Save model results
        log_message(f"\nSaving {MODEL_TYPE} results...")
        with open(f'{output_dir}/text/{MODEL_TYPE}_results.txt', 'w', encoding='utf-8') as f:
            f.write(f"{MODEL_TYPE.upper()} Model Results:\n\n")
            f.write(f"Mean ROC AUC: {np.mean(model_fold_metrics['roc_auc_scores']):.4f} (+/- {np.std(model_fold_metrics['roc_auc_scores']) * 2:.4f})\n")
            f.write(f"Mean pAUC (0-10% FPR): {np.mean([scores['pauc_01'] for scores in model_fold_metrics['pauc_scores']]):.4f} (+/- {np.std([scores['pauc_01'] for scores in model_fold_metrics['pauc_scores']]) * 2:.4f})\n")
            f.write(f"Mean Normalized pAUC (0-10% FPR): {np.mean(model_fold_metrics['pauc_01_scores']):.4f} (+/- {np.std(model_fold_metrics['pauc_01_scores']) * 2:.4f})\n")
            f.write(f"Mean pAUC (0-20% FPR): {np.mean([scores['pauc_02'] for scores in model_fold_metrics['pauc_scores']]):.4f} (+/- {np.std([scores['pauc_02'] for scores in model_fold_metrics['pauc_scores']]) * 2:.4f})\n")
            f.write(f"Mean Normalized pAUC (0-20% FPR): {np.mean(model_fold_metrics['pauc_02_scores']):.4f} (+/- {np.std(model_fold_metrics['pauc_02_scores']) * 2:.4f})\n")
            f.write(f"Median pAUC (0-10% FPR): {np.median([scores['pauc_01'] for scores in model_fold_metrics['pauc_scores']]):.4f}\n")
            f.write(f"Median Normalized pAUC (0-10% FPR): {np.median(model_fold_metrics['pauc_01_scores']):.4f}\n")
            f.write(f"Median pAUC (0-20% FPR): {np.median([scores['pauc_02'] for scores in model_fold_metrics['pauc_scores']]):.4f}\n")
            f.write(f"Median Normalized pAUC (0-20% FPR): {np.median(model_fold_metrics['pauc_02_scores']):.4f}\n\n")
            
            f.write("Fold-wise Results:\n")
            for fold, (pauc_scores, roc_auc) in enumerate(zip(
                model_fold_metrics['pauc_scores'],
                model_fold_metrics['roc_auc_scores']), 1):
                f.write(f"Fold {fold} - ROC AUC: {roc_auc:.4f}\n")
                f.write(f"  pAUC (0-10% FPR): {pauc_scores['pauc_01']:.4f}, Normalized: {pauc_scores['pauc_01_norm']:.4f}\n")
                f.write(f"  pAUC (0-20% FPR): {pauc_scores['pauc_02']:.4f}, Normalized: {pauc_scores['pauc_02_norm']:.4f}\n")
        
        # Plot model metrics
        log_message(f"Generating {MODEL_TYPE} plots...")
        plot_fold_metrics(model_fold_metrics, f'{output_dir}/{MODEL_TYPE}')
        
        # Save best model
        log_message(f"\nSaving best {MODEL_TYPE} model...")
        best_model_idx = np.argmax(model_fold_metrics['roc_auc_scores'])
        log_message(f"Selected best model from fold {best_model_idx + 1}")
        
        best_model = get_model(MODEL_TYPE, pretrained=False, device=device)
        log_message(f"Created new {MODEL_TYPE} model instance")
        
        # Log state dict information before loading
        log_message(f"Best model state_dict keys: {list(model_models[best_model_idx].keys())[:5]}...")
        best_model.load_state_dict(model_models[best_model_idx])
        log_message("Best model state_dict loaded successfully")
        
        # Save model checkpoint
        save_model_checkpoint(
            best_model,
            MODEL_TYPE,
            f'{output_dir}/models/best_{MODEL_TYPE}_model.pth'
        )
        
        model_end_time = datetime.now()
        model_duration = model_end_time - model_start_time
        log_message(f"\n{MODEL_TYPE} training and evaluation completed in {model_duration}")
        
        # Update models dictionary for ensemble
        models_dict = {
            MODEL_TYPE: best_model,
            'lgb': best_lgb_model,
            'xgb': best_xgb_model
        }
        
        # Evaluate models on test set if available
        if y_test is not None:
            log_message("\n=== Test Set Evaluation ===")
            test_start_time = datetime.now()
            
            # Load models if needed (in case they were saved and need to be reloaded)
            if isinstance(best_model, str):  # If it's a path instead of a model
                best_model = load_model_checkpoint(best_model, device)
                models_dict[MODEL_TYPE] = best_model
            
            # Get ensemble predictions
            all_predictions = get_ensemble_predictions(X_test, df_test, models_dict, device)
            
            # Calculate metrics for each model and ensemble
            metrics = {}
            for model_name, preds in all_predictions.items():
                metrics[model_name] = calculate_metrics(y_test, preds)
                
                # Plot confusion matrix for each model
                plot_confusion_matrix(
                    metrics[model_name]['confusion_matrix'],
                    model_name,
                    f'{output_dir}/plots/{model_name}_confusion_matrix.png'
                )
                
                # Log detailed metrics
                log_message(f"\n{model_name.upper()} Detailed Metrics:")
                log_message(f"ROC AUC: {metrics[model_name]['roc_auc']:.4f}")
                log_message(f"Classification Report:")
                for label, scores in metrics[model_name]['classification_report'].items():
                    if isinstance(scores, dict):
                        log_message(f"  Class {label}:")
                        for metric, value in scores.items():
                            log_message(f"    {metric}: {value:.4f}")
                
                # Save detailed metrics to file
                with open(f'{output_dir}/text/{model_name}_detailed_metrics.txt', 'w', encoding='utf-8') as f:
                    f.write(f"{model_name.upper()} Detailed Metrics:\n\n")
                    f.write(f"ROC AUC: {metrics[model_name]['roc_auc']:.4f}\n")
                    f.write("\nConfusion Matrix:\n")
                    f.write(str(metrics[model_name]['confusion_matrix']))
                    f.write("\n\nClassification Report:\n")
                    for label, scores in metrics[model_name]['classification_report'].items():
                        if isinstance(scores, dict):
                            f.write(f"\nClass {label}:\n")
                            for metric, value in scores.items():
                                f.write(f"  {metric}: {value:.4f}\n")
            
            # Plot metrics comparison across models (removed precision from comparison)
            plot_metrics_comparison(metrics, f'{output_dir}/plots/metrics_comparison.png')
            
            # Save ensemble results with new metrics
            with open(f'{output_dir}/text/ensemble_results.txt', 'w', encoding='utf-8') as f:
                f.write("Ensemble Model Results:\n\n")
                for model_name, model_metrics in metrics.items():
                    f.write(f"{model_name.upper()}:\n")
                    f.write(f"ROC AUC: {model_metrics['roc_auc']:.4f}\n")
                    f.write(f"Raw pAUC (0-10% FPR): {model_metrics['pauc_scores']['pauc_01']:.4f}\n")
                    f.write(f"Normalized pAUC (0-10% FPR): {model_metrics['pauc_scores']['pauc_01_norm']:.4f}\n")
                    f.write(f"Raw pAUC (0-20% FPR): {model_metrics['pauc_scores']['pauc_02']:.4f}\n")
                    f.write(f"Normalized pAUC (0-20% FPR): {model_metrics['pauc_scores']['pauc_02_norm']:.4f}\n")
                    f.write("\nConfusion Matrix:\n")
                    f.write(str(model_metrics['confusion_matrix']))
                    f.write("\n\nClassification Report:\n")
                    for label, scores in model_metrics['classification_report'].items():
                        if isinstance(scores, dict):
                            f.write(f"\nClass {label}:\n")
                            for metric, value in scores.items():
                                f.write(f"  {metric}: {value:.4f}\n")
                    f.write("\n" + "="*50 + "\n\n")
            
            # Plot ROC curves for all models
            plt.figure(figsize=(12, 8))
            for model_name, preds in all_predictions.items():
                fpr, tpr, _ = roc_curve(y_test, preds)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{model_name.upper()} (AUC = {roc_auc:.4f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves for All Models')
            plt.legend(loc="lower right")
            plt.savefig(f'{output_dir}/plots/ensemble_roc_curves.png')
            plt.close()
            
            test_end_time = datetime.now()
            test_duration = test_end_time - test_start_time
            log_message(f"\nTest set evaluation completed in {test_duration}")
        
        end_time = datetime.now()
        total_duration = end_time - start_time
        log_message("\n=== Pipeline Summary ===")
        log_message(f"Total pipeline duration: {total_duration}")
        log_message(f"LightGBM training duration: {lgb_duration}")
        log_message(f"XGBoost training duration: {xgb_duration}")
        log_message(f"{MODEL_TYPE} training duration: {model_duration}")
        if y_test is not None:
            log_message(f"Test set evaluation duration: {test_duration}")
        log_message(f"\nAll results have been saved to the '{output_dir}' directory")
        log_message("\n=== Pipeline completed successfully ===")
        
    except Exception as e:
        log_message(f"\nERROR: Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 