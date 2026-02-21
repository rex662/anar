import os
import json
import time
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from scipy import stats

# ==========================================
# CONFIGURATION - Optimized for DigitalOcean RTX 6000 (48GB VRAM)
# ==========================================
CONFIG = {
    "data_dir": "/Users/varnitkumar/Desktop/Anar/Pomegranate Diseases Dataset",  
    "output_dir": "/Users/varnitkumar/Desktop/Anar/Results",
    "batch_size": 256,           # Increased for 48GB VRAM
    "epochs": 20,
    "num_classes": 5,
    "seed": 42,
    "image_size": 224,
    "inception_size": 299,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    # Dataloader optimization for 8 vCPUs
    "num_workers": 6,            # Use 6 of 8 vCPUs for data loading
    "prefetch_factor": 4,        # Prefetch batches per worker
    "pin_memory": True,          # Faster CPU->GPU transfers
    "persistent_workers": True   # Keep workers alive between epochs
}

GLOBAL_METADATA = {
    "hardware_setup": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "gpu_count": torch.cuda.device_count(),
    "libraries": {"torch": torch.__version__},
    "models": {} 
}

# ==========================================
# 1. SETUP & UTILS
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enable cudnn.benchmark for fixed input sizes - significant speedup
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # Disable for better performance
        # Enable TF32 for RTX 6000 (Ampere arch) - faster matmul without mixed precision
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

set_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print GPU info for verification
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# ==========================================
# STATISTICAL ANALYSIS FUNCTIONS
# ==========================================
def bootstrap_confidence_interval(y_true, y_pred, metric_func, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence interval for a given metric.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        metric_func: Function that takes (y_true, y_pred) and returns a score
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level (default 95%)
    
    Returns:
        dict with point estimate, CI lower, CI upper, and standard error
    """
    n_samples = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Point estimate
    point_estimate = metric_func(y_true, y_pred)
    
    # Bootstrap sampling
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Calculate confidence interval (percentile method)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_scores, (alpha / 2) * 100)
    ci_upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)
    std_error = np.std(bootstrap_scores)
    
    return {
        "point_estimate": float(point_estimate),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "std_error": float(std_error),
        "confidence_level": confidence
    }

def mcnemar_test(y_true, y_pred_1, y_pred_2):
    """
    McNemar's test for comparing two classifiers.
    Tests if there's a significant difference between two models.
    
    Returns:
        dict with test statistic, p-value, and significance interpretation
    """
    y_true = np.array(y_true)
    y_pred_1 = np.array(y_pred_1)
    y_pred_2 = np.array(y_pred_2)
    
    # Build contingency table
    # b: model1 correct, model2 wrong
    # c: model1 wrong, model2 correct
    correct_1 = (y_pred_1 == y_true)
    correct_2 = (y_pred_2 == y_true)
    
    b = np.sum(correct_1 & ~correct_2)  # Model 1 correct, Model 2 wrong
    c = np.sum(~correct_1 & correct_2)  # Model 1 wrong, Model 2 correct
    
    # McNemar's test with continuity correction
    if b + c == 0:
        return {"statistic": 0, "p_value": 1.0, "significant": False, "interpretation": "Models are identical"}
    
    statistic = ((abs(b - c) - 1) ** 2) / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "interpretation": "Significant difference" if p_value < 0.05 else "No significant difference"
    }

def compute_all_confidence_intervals(y_true, y_pred, n_bootstrap=1000):
    """
    Compute confidence intervals for all standard metrics.
    """
    metrics = {
        "accuracy": lambda yt, yp: accuracy_score(yt, yp),
        "precision": lambda yt, yp: precision_score(yt, yp, average='weighted', zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, yp, average='weighted', zero_division=0),
        "f1": lambda yt, yp: f1_score(yt, yp, average='weighted', zero_division=0)
    }
    
    results = {}
    for name, func in metrics.items():
        results[name] = bootstrap_confidence_interval(y_true, y_pred, func, n_bootstrap)
    
    return results

def per_class_confidence_intervals(y_true, y_pred, classes, n_bootstrap=1000):
    """
    Compute per-class confidence intervals for precision, recall, and F1.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    results = {}
    
    for i, cls in enumerate(classes):
        # Binary conversion for this class
        yt_bin = (y_true == i).astype(int)
        yp_bin = (y_pred == i).astype(int)
        
        results[cls] = {
            "precision": bootstrap_confidence_interval(
                yt_bin, yp_bin,
                lambda yt, yp: precision_score(yt, yp, zero_division=0),
                n_bootstrap
            ),
            "recall": bootstrap_confidence_interval(
                yt_bin, yp_bin,
                lambda yt, yp: recall_score(yt, yp, zero_division=0),
                n_bootstrap
            ),
            "f1": bootstrap_confidence_interval(
                yt_bin, yp_bin,
                lambda yt, yp: f1_score(yt, yp, zero_division=0),
                n_bootstrap
            )
        }
    
    return results

# ==========================================
# 2. DATA LOADING (STRATIFIED SPLIT)
# ==========================================
# Transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'inception_train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'inception_val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Wrapper to apply transforms dynamically AFTER splitting
class TransformedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

# Load Data & Stratify
if not os.path.exists(CONFIG["data_dir"]):
    print(f"ERROR: Data path {CONFIG['data_dir']} does not exist.")
else:
    # 1. Load the full dataset (no transforms yet)
    full_dataset = datasets.ImageFolder(CONFIG["data_dir"])
    class_names = full_dataset.classes
    targets = full_dataset.targets
    print(f"Classes Detected: {class_names}")

    # 2. STRATIFIED SPLIT: 70% Train / 15% Val / 15% Test
    # First split: 70% train, 30% temp
    train_idx, temp_idx = train_test_split(
        np.arange(len(targets)),
        test_size=0.30,
        shuffle=True,
        stratify=targets, 
        random_state=CONFIG["seed"]
    )
    
    # Second split: 50% of temp (15% of total) for val, 50% for test
    temp_targets = [targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        shuffle=True,
        stratify=temp_targets,
        random_state=CONFIG["seed"]
    )

    # 3. Create Subsets
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    test_subset = Subset(full_dataset, test_idx)
    
    # 4. Verification Print
    print("\n" + "="*70)
    print("STRATIFIED SPLIT VERIFICATION (70/15/15)")
    print("="*70)
    train_counts = Counter([targets[i] for i in train_idx])
    val_counts = Counter([targets[i] for i in val_idx])
    test_counts = Counter([targets[i] for i in test_idx])
    
    print(f"{'Class':<20} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}")
    print("-" * 70)
    for i, cls in enumerate(class_names):
        t_c = train_counts[i]
        v_c = val_counts[i]
        te_c = test_counts[i]
        total = t_c + v_c + te_c
        print(f"{cls:<20} | {t_c:<8} | {v_c:<8} | {te_c:<8} | {total:<8}")
    print("-" * 70)
    print(f"{'TOTAL':<20} | {len(train_idx):<8} | {len(val_idx):<8} | {len(test_idx):<8} | {len(targets):<8}")
    print(f"{'PERCENTAGE':<20} | {100*len(train_idx)/len(targets):.1f}%    | {100*len(val_idx)/len(targets):.1f}%    | {100*len(test_idx)/len(targets):.1f}%    |")
    print("="*70 + "\n")

# ==========================================
# 3. MODEL FACTORY
# ==========================================
def get_model(model_name, num_classes):
    if model_name == "ResNet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, 'train'

    elif model_name == "VGG16":
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_classes)
        return model, 'train'

    elif model_name == "DenseNet121":
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model, 'train'

    elif model_name == "InceptionV3":
        model = models.inception_v3(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.aux_logits = False 
        return model, 'inception_train'

    elif model_name == "EfficientNetB0":
        try:
            model = models.efficientnet_b0(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        except AttributeError:
            import timm
            model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
        return model, 'train'
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: ResNet50, VGG16, DenseNet121, InceptionV3, EfficientNetB0")

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def calculate_metrics(y_true, y_pred, y_prob, classes):
    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    }
    class_wise = {}
    for i, cls in enumerate(classes):
        yt = [1 if y == i else 0 for y in y_true]
        yp = [1 if y == i else 0 for y in y_pred]
        class_wise[cls] = {
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0))
        }
    return overall, class_wise

def train_one_model(model_name):
    print(f"\n{'='*30}\nSTARTING: {model_name}\n{'='*30}")
    
    # Directories
    base_dir = os.path.join(CONFIG["output_dir"], model_name)
    for folder in ['weights', 'plots', 'data']:
        os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

    # Model & Config
    model, t_key = get_model(model_name, CONFIG["num_classes"])
    param_count = sum(p.numel() for p in model.parameters())
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Compile model for faster execution (PyTorch 2.0+)
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print(f"  [+] Model compiled with torch.compile()")
        except Exception as e:
            print(f"  [!] torch.compile not available: {e}")

    # Loaders with TransformedDataset wrapper
    # THIS is where augmentation happens (on train_subset)
    train_ds = TransformedDataset(train_subset, data_transforms[t_key])
    val_ds = TransformedDataset(val_subset, data_transforms[t_key.replace('train','val')])
    
    # Optimized DataLoaders for RTX 6000 + 8 vCPU + 64GB RAM
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        prefetch_factor=CONFIG["prefetch_factor"],
        persistent_workers=CONFIG["persistent_workers"],
        drop_last=True  # Avoid small final batch for consistent perf
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"] * 2,  # Larger batch for inference (no gradients)
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        prefetch_factor=CONFIG["prefetch_factor"],
        persistent_workers=CONFIG["persistent_workers"]
    )

    criterion = FocalLoss(gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    history = {"model": model_name, "history": []}
    best_f1 = 0.0
    start_time = time.time()

    # --- EPOCH LOOP ---
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        t_loss, t_preds, t_targs, t_samples = 0, [], [], 0
        
        # Train
        train_loop = tqdm(train_loader, desc=f"Ep {epoch}/{CONFIG['epochs']} [Tr]", leave=False)
        for inputs, labels in train_loop:
            # Non-blocking transfers for async CPU->GPU copy
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # More efficient gradient clearing
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            if isinstance(outputs, tuple): outputs = outputs[0]
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item() * inputs.size(0)
            t_samples += inputs.size(0)
            _, preds = torch.max(outputs, 1)
            t_preds.extend(preds.cpu().numpy())
            t_targs.extend(labels.cpu().numpy())
            train_loop.set_postfix(loss=loss.item())
            
        scheduler.step()
        
        # Validate
        model.eval()
        v_loss, v_preds, v_targs, v_probs, v_samples = 0, [], [], [], 0
        val_loop = tqdm(val_loader, desc=f"Ep {epoch}/{CONFIG['epochs']} [Val]", leave=False)
        
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * inputs.size(0)
                v_samples += inputs.size(0)
                
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                v_preds.extend(preds.cpu().numpy())
                v_targs.extend(labels.cpu().numpy())
                v_probs.extend(probs.cpu().numpy())
                val_loop.set_postfix(loss=loss.item())

        # Metrics
        t_ov, t_cw = calculate_metrics(t_targs, t_preds, None, class_names)
        v_ov, v_cw = calculate_metrics(v_targs, v_preds, v_probs, class_names)
        
        epoch_log = {
            "epoch": epoch,
            "train": {"loss": t_loss/t_samples, "metrics": t_ov, "class_wise": t_cw},
            "val": {"loss": v_loss/v_samples, "metrics": v_ov, "class_wise": v_cw}
        }
        history["history"].append(epoch_log)
        
        print(f"Epoch {epoch} | Val F1: {v_ov['f1']:.4f} | Loss: {epoch_log['val']['loss']:.4f}")

        if v_ov['f1'] > best_f1:
            best_f1 = v_ov['f1']
            torch.save(model.state_dict(), os.path.join(base_dir, 'weights', 'best_f1.pth'))

    # End Training
    total_training_time = time.time() - start_time
    torch.save(model.state_dict(), os.path.join(base_dir, 'weights', 'final.pth'))
    
    with open(os.path.join(base_dir, 'data', 'experiment_history.json'), 'w') as f:
        json.dump(history, f)

    # --- POST-TRAINING ANALYSIS ON HELD-OUT TEST SET ---
    print("--> Running Final Evaluation on TEST SET...")
    model.load_state_dict(torch.load(os.path.join(base_dir, 'weights', 'best_f1.pth')))
    model.eval()
    
    # Create test loader (unseen during training and validation)
    test_ds = TransformedDataset(test_subset, data_transforms[t_key.replace('train','val')])
    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG["batch_size"] * 2,
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        prefetch_factor=CONFIG["prefetch_factor"],
        persistent_workers=CONFIG["persistent_workers"]
    )

    # Latency
    dummy_size = 299 if "Inception" in model_name else 224
    dummy = torch.randn(1, 3, dummy_size, dummy_size).to(device)
    for _ in range(5): _ = model(dummy)
    lat_start = time.time()
    for _ in range(100):
        with torch.no_grad(): _ = model(dummy)
    latency_ms = ((time.time() - lat_start) / 100) * 1000

    # Final Metrics on TEST SET (no data leakage)
    final_targs, final_preds, final_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            out = model(inputs)
            prob = F.softmax(out, dim=1)
            final_probs.extend(prob.cpu().numpy())
            final_preds.extend(torch.argmax(prob, dim=1).cpu().numpy())
            final_targs.extend(labels.cpu().numpy())

    # Save CM
    cm = confusion_matrix(final_targs, final_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.savefig(os.path.join(base_dir, 'plots', 'confusion_matrix.png'))
    plt.close()
    
    with open(os.path.join(base_dir, 'data', 'confusion_matrix.json'), 'w') as f:
        json.dump({"matrix": cm.tolist(), "classes": class_names}, f)

    # Save ROC
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(final_targs, classes=range(len(class_names)))
    y_prob_np = np.array(final_probs)
    roc_data = {"classes": class_names, "curves": {}}
    
    plt.figure(figsize=(10,8))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob_np[:, i])
        auc_score = auc(fpr, tpr)
        roc_data["curves"][cls] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc_score)}
        plt.plot(fpr, tpr, label=f'{cls} (AUC={auc_score:.2f})')
    plt.plot([0,1],[0,1], 'k--')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'plots', 'roc_curve.png'))
    plt.close()
    
    with open(os.path.join(base_dir, 'data', 'roc_boundaries.json'), 'w') as f:
        json.dump(roc_data, f)

    # Statistical Analysis: Confidence Intervals
    print("--> Computing confidence intervals...")
    overall_ci = compute_all_confidence_intervals(final_targs, final_preds, n_bootstrap=1000)
    class_ci = per_class_confidence_intervals(final_targs, final_preds, class_names, n_bootstrap=1000)
    
    statistical_results = {
        "overall_metrics_with_ci": overall_ci,
        "per_class_metrics_with_ci": class_ci,
        "n_bootstrap": 1000,
        "confidence_level": 0.95
    }
    
    with open(os.path.join(base_dir, 'data', 'statistical_analysis.json'), 'w') as f:
        json.dump(statistical_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"STATISTICAL SUMMARY: {model_name}")
    print(f"{'='*50}")
    for metric, ci_data in overall_ci.items():
        print(f"{metric.upper():>12}: {ci_data['point_estimate']:.4f} "
              f"(95% CI: [{ci_data['ci_lower']:.4f}, {ci_data['ci_upper']:.4f}])")
    print(f"{'='*50}\n")
    
    # Store predictions for cross-model comparison
    GLOBAL_METADATA["models"][model_name] = {
        "params": param_count,
        "inference_latency_ms": latency_ms,
        "training_time_sec": total_training_time,
        "best_val_f1": best_f1,
        "statistical_ci": overall_ci,
        "predictions": final_preds,
        "ground_truth": final_targs
    }

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    model_list = ["ResNet50", "InceptionV3", "DenseNet121", "EfficientNetB0", "VGG16"]
    
    for m_name in model_list:
        try:
            train_one_model(m_name)
            # Clear CUDA cache between models to prevent memory fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            print(f"FAILED {m_name}: {e}")
            import traceback
            traceback.print_exc()

    with open(os.path.join(CONFIG["output_dir"], "00_System_Metadata.json"), 'w') as f:
        # Remove predictions from metadata before saving (too large)
        metadata_to_save = copy.deepcopy(GLOBAL_METADATA)
        for model_name in metadata_to_save["models"]:
            metadata_to_save["models"][model_name].pop("predictions", None)
            metadata_to_save["models"][model_name].pop("ground_truth", None)
        json.dump(metadata_to_save, f, indent=4)
    
    # Cross-model statistical significance tests
    print("\n" + "="*60)
    print("CROSS-MODEL STATISTICAL SIGNIFICANCE TESTS (McNemar's Test)")
    print("="*60)
    
    model_names = list(GLOBAL_METADATA["models"].keys())
    significance_results = {}
    
    if len(model_names) >= 2:
        ground_truth = GLOBAL_METADATA["models"][model_names[0]]["ground_truth"]
        
        for i, model_a in enumerate(model_names):
            for model_b in model_names[i+1:]:
                preds_a = GLOBAL_METADATA["models"][model_a]["predictions"]
                preds_b = GLOBAL_METADATA["models"][model_b]["predictions"]
                
                test_result = mcnemar_test(ground_truth, preds_a, preds_b)
                comparison_key = f"{model_a}_vs_{model_b}"
                significance_results[comparison_key] = test_result
                
                print(f"{model_a} vs {model_b}:")
                print(f"  p-value: {test_result['p_value']:.6f}")
                print(f"  Result: {test_result['interpretation']}")
                print()
        
        # Save significance test results
        with open(os.path.join(CONFIG["output_dir"], "00_Significance_Tests.json"), 'w') as f:
            json.dump(significance_results, f, indent=4)
        
        print(f"Significance tests saved to: {CONFIG['output_dir']}/00_Significance_Tests.json")
    
    print("="*60)
        
    print("\nALL TRAINING COMPLETE.")