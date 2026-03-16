"""
PAD-UFES-20 Dataset Description

Summary:
The PAD-UFES-20 dataset contains skin lesion images and metadata collected from 1,373 patients, 
consisting of 1,641 skin lesions and 2,298 images. The dataset includes six types of skin lesions:
- Skin Cancers: Basal Cell Carcinoma (BCC), Melanoma (MEL), and Squamous Cell Carcinoma (SCC)
- Skin Diseases: Actinic Keratosis (ACK), Nevus (NEV), and Seborrheic Keratosis (SEK)

Key Points:
- All BCC, SCC, and MEL cases are biopsy-proven
- Approximately 58% of samples are biopsy-proven
- Images are collected using different smartphone devices and are in PNG format
- Each image has associated metadata with 26 features
- Bowen's disease (BOD) is considered SCC in situ and is clustered with SCC

Dataset Statistics:
- Number of patients: 1,373
- Number of lesions: 1,641
- Number of images: 2,298
- Number of features: 26

Feature Descriptions:
1. patient_id: Patient identifier
2. lesion_id: Lesion/wound identifier
3. smoke: Smoking history
4. drink: Alcohol consumption history
5. background_father: Father's disease history
6. background_mother: Mother's disease history
7. age: Patient age at examination
8. pesticide: Pesticide/chemical exposure
9. gender: Patient gender
10. skin_cancer_history: Family skin cancer history
11. cancer_history: Family cancer history
12. has_piped_water: Access to piped water
13. has_sewage_system: Access to sewage system
14. fitspatrick: Skin sun tolerance
15. region: Body area of lesion
16. diameter_1: Primary lesion diameter
17. diameter_2: Secondary lesion diameter
18. diagnostic: Lesion diagnosis type
19. itch: Lesion itching
20. grew: Lesion growth
21. hurt: Lesion pain
22. changed: Lesion appearance change
23. bleed: Lesion bleeding
24. elevation: Lesion elevation relative to skin
25. img_id: Image identifier
26. biopsed: Biopsy status
"""

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
DATASET_DIR = './dataset/PAD-UFES-20'
TRAIN_METADATA_PATH = 'V:/NCKH/dataset/PAD-UFES-20/pad_train_metadata.csv'
TEST_METADATA_PATH = 'V:/NCKH/dataset/PAD-UFES-20/pad_test_metadata.csv'
IMG_DIRS = [
    os.path.join(DATASET_DIR, 'imgs_part_1', 'imgs_part_1'),
    os.path.join(DATASET_DIR, 'imgs_part_2', 'imgs_part_2'),
    os.path.join(DATASET_DIR, 'imgs_part_3', 'imgs_part_3')
]

# Model selection
MODEL_TYPE = 'hybrid'  # Options: 'efficientnet_b0', 'vit_base_patch16_224', 'hybrid'

# Model parameters
SEED = 42
ERR = 1e-5  
USE_SMOTE = False  # Toggle SMOTE on/off
SAMPLING_RATIO = 0.5  
K_NEIGHBORS = 5      
N_SPLITS = 3
EARLY_STOPPING_ROUNDS = 4

# Training parameters
NUM_WORKERS = 2  # Number of workers for data loading
PIN_MEMORY = True  # Pin memory for faster data transfer to GPU

# Column names
ID_COL = 'img_id'
GROUP_COL = 'patient_id'
TARGET_COL = 'diagnostic'  # Multi-class classification

# Image model parameters (for ViT, EfficientNet, and Hybrid)
IMG_MODEL_BATCH_SIZE = 32
IMG_MODEL_NUM_EPOCHS = 30
IMG_MODEL_LEARNING_RATE = 1e-5
IMG_MODEL_WEIGHT_DECAY = 1e-4
IMG_MODEL_EARLY_STOPPING_ROUNDS = 4

# Ensemble weights
ENSEMBLE_WEIGHTS = {
    MODEL_TYPE: 0.4,  
    'lgb': 0.3,        
    'xgb': 0.3          
}

# Feature columns for PAD dataset
NUM_COLS = [
    'age',
    'diameter_1',
    'diameter_2',
    'itch',
    'grew',
    'hurt',
    'changed',
    'bleed',
    'elevation',
    'biopsed'
]

CAT_COLS = [
    'smoke',
    'drink',
    'background_father',
    'background_mother',
    'pesticide',
    'gender',
    'skin_cancer_history',
    'cancer_history',
    'has_piped_water',
    'has_sewage_system',
    'fitspatrick',
    'region'
]

FEATURE_COLS = NUM_COLS + CAT_COLS

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

# Custom dataset for image loading
class PADDataset(Dataset):
    def __init__(self, df, transform=None, is_train=True):
        self.df = df
        self.img_dirs = IMG_DIRS
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx][ID_COL]
        
        # Try to find the image in each directory
        image = None
        for img_dir in self.img_dirs:
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert('RGB')
                    break
                except:
                    continue
        
        if image is None:
            # If image loading fails, return a black image
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_train:
            target = self.df.iloc[idx][TARGET_COL]
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

def read_data(path):
    dtypes = {
        'age': 'float64',
        'diameter_1': 'float64',
        'diameter_2': 'float64',
        'itch': 'object',
        'grew': 'object',
        'hurt': 'object',
        'changed': 'object',
        'bleed': 'object',
        'elevation': 'object',
        'biopsed': 'object'
    }
    df = pd.read_csv(path, low_memory=False, dtype=dtypes)
    # Chuyển TRUE/FALSE về 1/0 cho toàn bộ DataFrame
    df = df.replace({'TRUE': 1, 'FALSE': 0, 'True': 1, 'False': 0})
    # Đảm bảo các cột số đúng kiểu
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['age'] = df['age'].fillna(df['age'].median())
    for col in CAT_COLS:
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)
    return df

def preprocess(df_train, df_test):
    global CAT_COLS
    
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32, handle_unknown='ignore')
    encoder.fit(df_train[CAT_COLS])
    
    new_cat_cols = [f'onehot_{i}' for i in range(len(encoder.get_feature_names_out()))]

    # Convert one-hot encoded features to integers
    df_train[new_cat_cols] = encoder.transform(df_train[CAT_COLS]).astype(int)
    df_test[new_cat_cols] = encoder.transform(df_test[CAT_COLS]).astype(int)

    # Update feature columns
    for col in CAT_COLS:
        if col in FEATURE_COLS:
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
        if model_name == MODEL_TYPE:  # Skip the hybrid model
            continue
        if model_name == 'cat':  # Skip CatBoost predictions
            continue
        if model_name == 'lgb':
            preds = model.predict(test_data, raw_score=True)
            metadata_preds[model_name] = 1 / (1 + np.exp(-preds))
        elif model_name == 'xgb':
            metadata_preds[model_name] = model.predict_proba(test_data)[:, 1]
    
    # Get image or hybrid model predictions
    if MODEL_TYPE == 'hybrid':
        test_dataset = HybridDataset(
            test_images,
            FEATURE_COLS,
            transform=get_transforms('val'),
            is_train=False
        )
    else:
        test_dataset = PADDataset(
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
    
    model = models_dict[MODEL_TYPE]
    model.eval()
    predictions = []
    
    with torch.no_grad():
        if MODEL_TYPE == 'hybrid':
            for images, metadata in test_loader:
                images = images.to(device)
                metadata = metadata.to(device)
                outputs = model(images, metadata).squeeze()
                predictions.extend(torch.sigmoid(outputs).cpu().numpy())
        else:
            for images in test_loader:
                images = images.to(device)
                outputs = model(images).squeeze()
                predictions.extend(torch.sigmoid(outputs).cpu().numpy())
    
    predictions = np.array(predictions)
    
    # Combine all predictions
    ensemble_pred = ensemble_predictions(predictions, metadata_preds)
    
    return {
        'ensemble': ensemble_pred,
        MODEL_TYPE: predictions,
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
    # Handle NaN values in predictions
    y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5)
    
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

class HybridModel(nn.Module):
    def __init__(self, model_name, num_metadata_features, pretrained=True):
        super(HybridModel, self).__init__()

        # EfficientNet branch
        self.efficientnet = timm.create_model('tf_efficientnet_b0', pretrained=pretrained)
        for param in list(self.efficientnet.parameters())[:-4]:
            param.requires_grad = False
        eff_dim = self.efficientnet.classifier.in_features  # Should be 1280

        # ViT branch
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        for param in list(self.vit.parameters())[:-4]:
            param.requires_grad = False
        vit_dim = self.vit.head.in_features  # Should be 768

        # Metadata processing branch
        self.metadata_layers = nn.Sequential(
            nn.LayerNorm(num_metadata_features),
            nn.Linear(num_metadata_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Projection to unify feature dimensions to 256 each
        self.eff_proj = nn.Sequential(
            nn.Linear(eff_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Gating mechanism to combine features
        self.gate_fc = nn.Sequential(
            nn.Linear(512, 2),  # 512 = 256 + 256 (eff_proj + vit_proj)
            nn.Softmax(dim=1)
        )

        # Fusion block with residual enhancement and metadata integration
        # Input dim: 512 (vision_fused) + 128 (metadata) = 640
        self.fusion = nn.Sequential(
            nn.Linear(640, 512),  # 640 = 512 (vision_fused) + 128 (metadata)
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 1)  # Binary classification
        )

        # Auxiliary classifiers
        self.aux_eff = nn.Linear(256, 1)  # Changed from 512 to 256
        self.aux_vit = nn.Linear(256, 1)  # Changed from 512 to 256
        self.aux_metadata = nn.Linear(128, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, images, metadata):
        # Process metadata
        metadata = torch.nan_to_num(metadata, nan=0.0)
        metadata_features = self.metadata_layers(metadata)  # Output: 128

        # EfficientNet path
        eff_feat = self.efficientnet.forward_features(images)
        eff_feat = self.efficientnet.global_pool(eff_feat).flatten(1)
        eff_feat = self.eff_proj(eff_feat)  # Output: 256

        # ViT path
        vit_tokens = self.vit.forward_features(images)
        cls_token = vit_tokens[:, 0]
        patch_token = vit_tokens[:, 1:].mean(dim=1)
        vit_feat = cls_token + patch_token
        vit_feat = self.vit_proj(vit_feat)  # Output: 256

        # Gating weights for vision features
        gate_input = torch.cat([eff_feat, vit_feat], dim=1)  # 512 = 256 + 256
        gate_scores = self.gate_fc(gate_input)
        eff_gate, vit_gate = gate_scores[:, 0:1], gate_scores[:, 1:2]

        # Weighted combination of vision features
        vision_fused = torch.cat([eff_feat * eff_gate, vit_feat * vit_gate], dim=1)  # Output: 512

        # Combine with metadata features
        combined = torch.cat([vision_fused, metadata_features], dim=1)  # 640 = 512 + 128
        
        # Final fusion and prediction
        output = self.fusion(combined)

        if self.training:
            aux_eff = self.aux_eff(eff_feat)
            aux_vit = self.aux_vit(vit_feat)
            aux_metadata = self.aux_metadata(metadata_features)
            return output, aux_eff, aux_vit, aux_metadata
        
        return output

class HybridDataset(Dataset):
    def __init__(self, df, metadata_features, transform=None, is_train=True):
        self.df = df
        self.img_dirs = IMG_DIRS
        self.transform = transform
        self.is_train = is_train
        self.metadata_features = metadata_features
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx][ID_COL]
        
        # Load image
        image = None
        for img_dir in self.img_dirs:
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert('RGB')
                    break
                except:
                    continue
        
        if image is None:
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        # Get metadata features
        metadata = torch.tensor(self.df.iloc[idx][self.metadata_features].values.astype(np.float32))
        
        if self.is_train:
            target = self.df.iloc[idx][TARGET_COL]
            return image, metadata, target
        return image, metadata

def train_one_epoch(model, loader, optimizer, criterion, device, is_hybrid=False):
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    pbar = tqdm(loader, total=len(loader))
    for batch_idx, batch_data in enumerate(pbar):
        if is_hybrid:
            images, metadata, target = batch_data
            images = images.to(device)
            metadata = metadata.to(device)
            target = target.to(device)
            
            # Check for NaN in metadata
            if torch.isnan(metadata).any():
                metadata = torch.nan_to_num(metadata, nan=0.0)
            
            outputs = model(images, metadata)
            if isinstance(outputs, tuple):
                main_output, aux_eff, aux_vit, aux_metadata = outputs
                # Calculate main loss and auxiliary losses
                main_loss = criterion(main_output.squeeze(), target.float())
                aux_eff_loss = criterion(aux_eff.squeeze(), target.float())
                aux_vit_loss = criterion(aux_vit.squeeze(), target.float())
                aux_metadata_loss = criterion(aux_metadata.squeeze(), target.float())
                
                # Combine losses with weights
                loss = main_loss + 0.3 * (aux_eff_loss + aux_vit_loss + aux_metadata_loss)
                output = main_output  # Use main output for metrics
            else:
                output = outputs
                loss = criterion(output.squeeze(), target.float())
        else:
            data, target = batch_data
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output.squeeze(), target.float())
        
        optimizer.zero_grad()
        
        # Check if loss is NaN
        if torch.isnan(loss):
            print(f"Warning: NaN loss detected at batch {batch_idx}")
            continue
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Convert predictions to numpy and handle NaN
        batch_preds = output.squeeze().detach().cpu().numpy()
        batch_preds = np.nan_to_num(batch_preds, nan=0.5)  # Replace NaN with 0.5
        predictions.extend(batch_preds)
        targets.extend(target.cpu().numpy())
        
        pbar.set_description(f'Loss: {loss.item():.4f}')
    
    # Handle case where all predictions are the same value
    if len(set(predictions)) == 1:
        print("Warning: All predictions are the same value")
        return total_loss / len(loader), 0.5
    
    try:
        auc = roc_auc_score(targets, predictions)
    except ValueError as e:
        print(f"Warning: Could not calculate AUC: {str(e)}")
        auc = 0.5
    
    return total_loss / len(loader), auc

def validate(model, loader, criterion, device, is_hybrid=False):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_data in loader:
            if is_hybrid:
                images, metadata, target = batch_data
                images = images.to(device)
                metadata = metadata.to(device)
                target = target.to(device)
                
                # Check for NaN in metadata
                if torch.isnan(metadata).any():
                    metadata = torch.nan_to_num(metadata, nan=0.0)
                
                outputs = model(images, metadata)
                if isinstance(outputs, tuple):
                    main_output, aux_eff, aux_vit, aux_metadata = outputs
                    # Calculate main loss and auxiliary losses
                    main_loss = criterion(main_output.squeeze(), target.float())
                    aux_eff_loss = criterion(aux_eff.squeeze(), target.float())
                    aux_vit_loss = criterion(aux_vit.squeeze(), target.float())
                    aux_metadata_loss = criterion(aux_metadata.squeeze(), target.float())
                    
                    # Combine losses with weights
                    loss = main_loss + 0.3 * (aux_eff_loss + aux_vit_loss + aux_metadata_loss)
                    output = main_output  # Use main output for metrics
                else:
                    output = outputs
                    loss = criterion(output.squeeze(), target.float())
            else:
                data, target = batch_data
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                loss = criterion(output.squeeze(), target.float())
            
            if not torch.isnan(loss):
                total_loss += loss.item()
            
            # Convert predictions to numpy and handle NaN
            batch_preds = output.squeeze().cpu().numpy()
            batch_preds = np.nan_to_num(batch_preds, nan=0.5)
            predictions.extend(batch_preds)
            targets.extend(target.cpu().numpy())
    
    try:
        auc = roc_auc_score(targets, predictions)
    except ValueError as e:
        print(f"Warning: Could not calculate AUC: {str(e)}")
        auc = 0.5
    
    return total_loss / len(loader), auc

def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, is_hybrid=False):
    best_val_auc = 0
    best_model = None
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=2, verbose=True
    )
    
    # Initialize scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, device, is_hybrid)
        print(f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}')
        
        val_loss, val_auc = validate(model, val_loader, criterion, device, is_hybrid)
        print(f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
        
        # Only update scheduler if val_auc is valid
        if not np.isnan(val_auc):
            scheduler.step(val_auc)
        
        if val_auc > best_val_auc and not np.isnan(val_auc):
            best_val_auc = val_auc
            best_model = model.state_dict().copy()
            print(f'New best model saved! Val AUC: {val_auc:.4f}')
    
    return model, best_model, best_val_auc

class PADModel(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        
        # Initialize the base model
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        # Get number of features based on model architecture
        if 'efficientnet' in model_name:
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif 'vit' in model_name:
            n_features = self.model.head.in_features
            self.model.head = nn.Identity()
        else:
            raise ValueError(f"Model {model_name} not supported. Use 'efficientnet_b0' or 'vit_base_patch16_224'")
        
        # Custom classifier with dropout and batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 1024),
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
            
            nn.Linear(256, 1)  # Binary classification
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features from base model
        features = self.model(x)
        
        # Pass through classifier
        out = self.classifier(features)
        return out

def main():
    try:
        global LOG_FILE
        start_time = datetime.now()
        
        # Create output directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'out/dacs_pad_{timestamp}_{MODEL_TYPE}_ensemble_img_metadata'
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
        
        # Chuyển nhãn diagnostic thành nhị phân: MEL, BCC, SCC là 1, còn lại là 0
        cancer_labels = ['MEL', 'BCC', 'SCC']
        df_train[TARGET_COL] = df_train[TARGET_COL].apply(lambda x: 1 if x in cancer_labels else 0)
        if TARGET_COL in df_test.columns:
            df_test[TARGET_COL] = df_test[TARGET_COL].apply(lambda x: 1 if x in cancer_labels else 0)
        
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
            if MODEL_TYPE == 'hybrid':
                model = HybridModel('efficientnet_b0', len(FEATURE_COLS), pretrained=True).to(device)
            else:
                model = PADModel(MODEL_TYPE, pretrained=True).to(device)
            
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.AdamW(model.parameters(), lr=IMG_MODEL_LEARNING_RATE, weight_decay=IMG_MODEL_WEIGHT_DECAY)
            
            # Create datasets
            if MODEL_TYPE == 'hybrid':
                train_dataset = HybridDataset(
                    df_train.iloc[train_idx],
                    FEATURE_COLS,
                    transform=get_transforms('train'),
                    is_train=True
                )
                val_dataset = HybridDataset(
                    df_train.iloc[val_idx],
                    FEATURE_COLS,
                    transform=get_transforms('val'),
                    is_train=True
                )
            else:
                train_dataset = PADDataset(
                    df_train.iloc[train_idx],
                    transform=get_transforms('train'),
                    is_train=True
                )
                val_dataset = PADDataset(
                    df_train.iloc[val_idx],
                    transform=get_transforms('val'),
                    is_train=True
                )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=IMG_MODEL_BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=IMG_MODEL_BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY
            )
            
            # Train model
            model, best_model_state, val_auc = train_model(
                model, train_loader, val_loader, optimizer, criterion, device, IMG_MODEL_NUM_EPOCHS, MODEL_TYPE == 'hybrid'
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
                if MODEL_TYPE == 'hybrid':
                    for images, metadata, _ in val_loader:
                        images = images.to(device)
                        metadata = metadata.to(device)
                        
                        # Handle NaN in metadata
                        metadata = torch.nan_to_num(metadata, nan=0.0)
                        
                        outputs = model(images, metadata).squeeze()
                        # Handle NaN in outputs
                        outputs = torch.nan_to_num(outputs, nan=0.0)
                        val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                else:
                    for images, _ in val_loader:
                        images = images.to(device)
                        outputs = model(images).squeeze()
                        # Handle NaN in outputs
                        outputs = torch.nan_to_num(outputs, nan=0.0)
                        val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            
            val_preds = np.array(val_preds)
            # Final NaN check on entire predictions array
            val_preds = np.nan_to_num(val_preds, nan=0.5)
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
        
        if MODEL_TYPE == 'hybrid':
            best_model = HybridModel('efficientnet_b0', len(FEATURE_COLS), pretrained=False).to(device)
        else:
            best_model = PADModel(MODEL_TYPE, pretrained=False).to(device)
        
        log_message(f"Created new {MODEL_TYPE} model instance")
        
        # Log state dict information before loading
        log_message(f"Best model state_dict keys: {list(model_models[best_model_idx].keys())[:5]}...")
        best_model.load_state_dict(model_models[best_model_idx])
        log_message("Best model state_dict loaded successfully")
        
        # Save model checkpoint
        torch.save(best_model.state_dict(), f'{output_dir}/models/best_{MODEL_TYPE}_model.pth')
        
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
                best_model = torch.load(best_model)
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