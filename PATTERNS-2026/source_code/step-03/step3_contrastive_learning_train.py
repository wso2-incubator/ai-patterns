"""
Supervised Contrastive Learning pipeline: 5-fold cross-validation and final model training.

Hardware Requirement:
    Fine-tuning the default 8B-parameter backbone (BAAI/bge-reasoner-embed-qwen3-8b-0923)
    requires > 50 GB GPU VRAM. An NVIDIA A100 (80GB) GPU or equivalent is recommended.
"""

from __future__ import annotations

import gc
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# -------------------------- Config --------------------------
CONFIG = {
    "dataset_path": "/datasets/labeled_data.csv",
    "min_samples_per_label": 20,
    "model_name": "BAAI/bge-reasoner-embed-qwen3-8b-0923",  # Or "BAAI/bge-code-v1"
    "hn_base_model": "google-bert/bert-base-uncased",
    "loss_margin": 0.5,
    "use_hard_negatives": True,
    "num_hard_negatives": 10,
    "max_pairs_per_class": 80,
    "epochs": 3,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "warmup_steps": 10,
    "max_seq_length": 768,
    "seed": 42,
    "n_splits": 5,
    "knn_neighbors": 15,
    "output_model_dir": "./saved_models/final_model",
}


# -------------------------- Utilities --------------------------
def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def clear_memory() -> None:
    """Trigger garbage collection and release cached CUDA memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -------------------------- Preprocessing --------------------------
def load_and_preprocess_data(
    filepath: str,
    min_samples_per_label: int = 20,
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Load dataset from CSV, filter classes below threshold, and encode target labels.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    df = pd.read_csv(filepath)
    label_counts = df["label"].value_counts()
    valid_labels = label_counts[label_counts >= min_samples_per_label].index
    filtered_df = df[df["label"].isin(valid_labels)].reset_index(drop=True)

    le = LabelEncoder()
    filtered_df["label_enc"] = le.fit_transform(filtered_df["label"])
    return filtered_df, le


# -------------------------- Hard Negative Mining --------------------------
def extract_base_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Generate normalized embeddings using a lightweight base model for hard negative mining.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = SentenceTransformer(model_name, device=device)
    embeddings = base_model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    del base_model
    clear_memory()
    return embeddings


def mine_hard_negatives(
    labels: np.ndarray,
    embeddings: np.ndarray,
    max_candidates: int = 50,
) -> Dict[int, List[int]]:
    """
    Find top-k hardest negatives (highest cosine similarity from different classes) for each anchor sample.
    """
    sim_matrix = embeddings @ embeddings.T
    n_samples = len(labels)
    hard_negatives = {}

    for idx in range(n_samples):
        anchor_label = labels[idx]
        diff_class_indices = np.where(labels != anchor_label)[0]
        if len(diff_class_indices) == 0:
            hard_negatives[idx] = []
            continue

        sims = sim_matrix[idx, diff_class_indices]
        k = min(max_candidates, len(diff_class_indices))
        top_k_indices = np.argsort(sims)[-k:][::-1]
        hard_negatives[idx] = diff_class_indices[top_k_indices].tolist()

    return hard_negatives


def build_contrastive_pairs(
    texts: List[str],
    labels: List[int],
    embeddings: Optional[np.ndarray] = None,
    max_pairs_per_class: int = 80,
    num_hard_negatives: int = 10,
    use_hard_negatives: bool = True,
) -> List[InputExample]:
    """
    Construct positive (intra-class, label=1.0) and negative (inter-class, label=0.0) pairs for contrastive loss.
    """
    label_to_indices = {}
    for idx, lbl in enumerate(labels):
        label_to_indices.setdefault(lbl, []).append(idx)

    examples = []

    # Intra-class positive pairs (label = 1.0)
    for lbl, indices in label_to_indices.items():
        if len(indices) < 2:
            continue
        pairs = [(indices[i], indices[j]) for i in range(len(indices)) for j in range(i + 1, len(indices))]
        if max_pairs_per_class and len(pairs) > max_pairs_per_class:
            pairs = random.sample(pairs, max_pairs_per_class)
        for a, b in pairs:
            examples.append(InputExample(texts=[str(texts[a]), str(texts[b])], label=1.0))

    # Inter-class negative pairs (label = 0.0)
    labels_arr = np.array(labels)
    if use_hard_negatives and embeddings is not None:
        hard_negs_map = mine_hard_negatives(labels_arr, embeddings, max_candidates=50)
        for lbl, indices in label_to_indices.items():
            used_negatives_for_class = set()
            for anchor_idx in indices:
                candidates = hard_negs_map.get(anchor_idx, [])
                chosen = []
                for neg_idx in candidates:
                    if neg_idx not in used_negatives_for_class:
                        chosen.append(neg_idx)
                        if len(chosen) >= num_hard_negatives:
                            break
                for neg_idx in chosen:
                    examples.append(InputExample(texts=[str(texts[anchor_idx]), str(texts[neg_idx])], label=0.0))
                    used_negatives_for_class.add(neg_idx)
    else:
        all_indices = list(range(len(texts)))
        for anchor_idx in all_indices:
            neg_candidates = [i for i in all_indices if labels_arr[i] != labels_arr[anchor_idx]]
            if neg_candidates:
                chosen_negs = random.sample(neg_candidates, min(num_hard_negatives, len(neg_candidates)))
                for neg_idx in chosen_negs:
                    examples.append(InputExample(texts=[str(texts[anchor_idx]), str(texts[neg_idx])], label=0.0))

    random.shuffle(examples)
    return examples


# -------------------------- Model Training --------------------------
def train_contrastive_model(
    model_name: str,
    train_examples: List[InputExample],
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    warmup_steps: int = 10,
    max_seq_length: int = 768,
    loss_margin: float = 0.5,
) -> SentenceTransformer:
    """
    Fine-tune a SentenceTransformer model using ContrastiveLoss on positive/negative pairs.
    """
    clear_memory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    model.max_seq_length = max_seq_length

    use_bf16 = torch.cuda.is_available()
    if use_bf16:
        model = model.to(torch.bfloat16)

    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=model.smart_batching_collate,
    )

    loss_fn = losses.ContrastiveLoss(model=model, margin=loss_margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.fit(
        train_objectives=[(train_loader, loss_fn)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_class=type(optimizer),
        optimizer_params={"lr": lr, "weight_decay": 0.01},
        show_progress_bar=True,
        use_amp=not use_bf16,
    )
    return model


# -------------------------- Centroid & Classifiers --------------------------
def build_centroids(
    embeddings: torch.Tensor,
    labels: List[int],
) -> Dict[int, torch.Tensor]:
    """
    Compute L2-normalized mean class embedding vectors (centroids) for each class.
    """
    centroids = {}
    labels_arr = np.array(labels)
    for class_id in np.unique(labels_arr):
        class_embs = embeddings[labels_arr == class_id]
        mean_vec = class_embs.mean(dim=0)
        centroids[int(class_id)] = F.normalize(mean_vec, dim=-1).cpu()
    return centroids


def predict_centroid(
    embeddings: torch.Tensor,
    centroids: Dict[int, torch.Tensor],
) -> Tuple[List[int], np.ndarray]:
    """
    Classify embeddings against class centroids via cosine similarity and return predictions with confidence scores.
    """
    classes = sorted(centroids.keys())
    centroid_matrix = torch.stack([centroids[c] for c in classes], dim=0).to(embeddings.device).float()
    sims = (embeddings.float() @ centroid_matrix.T).cpu().numpy()
    pred_indices = sims.argmax(axis=1)
    confidence_scores = sims.max(axis=1)
    predicted_labels = [classes[i] for i in pred_indices]
    return predicted_labels, confidence_scores


def predict_knn(
    train_embs: torch.Tensor,
    train_labels: List[int],
    test_embs: torch.Tensor,
    n_neighbors: int = 15,
) -> List[int]:
    """
    Classify test embeddings using K-Nearest Neighbors with cosine metric.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine", n_jobs=-1)
    knn.fit(train_embs.cpu().numpy(), train_labels)
    return knn.predict(test_embs.cpu().numpy()).tolist()


# -------------------------- Cross-Validation --------------------------
def run_5fold_cross_validation(
    df: pd.DataFrame,
    class_names: List[str],
    config: Dict[str, Any] = CONFIG,
) -> None:
    """
    Run 5-fold Stratified Group Cross-Validation to evaluate contrastive model performance out-of-fold.
    """
    print("\n" + "=" * 80)
    print("STARTING 5-FOLD STRATIFIED GROUP CROSS-VALIDATION")
    print("=" * 80)

    texts = df["code_summary"].values
    labels = df["label_enc"].values
    groups = df["file"].values if "file" in df.columns else None

    if groups is not None:
        print(f"Grouped by unique files: {len(np.unique(groups))} files.")
        sgkf = StratifiedGroupKFold(n_splits=config["n_splits"], shuffle=True, random_state=config["seed"])
        splits = list(sgkf.split(texts, labels, groups=groups))
    else:
        skf = StratifiedKFold(n_splits=config["n_splits"], shuffle=True, random_state=config["seed"])
        splits = list(skf.split(texts, labels))

    all_true = []
    all_pred_centroid = []
    all_pred_knn = []
    fold_accuracies_centroid = []
    fold_f1_centroid = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\n--- FOLD {fold}/{config['n_splits']} ---")
        X_train, y_train = texts[train_idx].tolist(), labels[train_idx].tolist()
        X_test, y_test = texts[test_idx].tolist(), labels[test_idx].tolist()

        train_embeddings = None
        if config["use_hard_negatives"]:
            print(f"Mining hard negatives using {config['hn_base_model']}...")
            train_embeddings = extract_base_embeddings(
                X_train,
                model_name=config["hn_base_model"],
                batch_size=config["batch_size"],
            )

        train_examples = build_contrastive_pairs(
            texts=X_train,
            labels=y_train,
            embeddings=train_embeddings,
            max_pairs_per_class=config["max_pairs_per_class"],
            num_hard_negatives=config["num_hard_negatives"],
            use_hard_negatives=config["use_hard_negatives"],
        )
        print(f"Constructed {len(train_examples)} contrastive training pairs.")

        fold_model = train_contrastive_model(
            model_name=config["model_name"],
            train_examples=train_examples,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr=config["learning_rate"],
            warmup_steps=config["warmup_steps"],
            max_seq_length=config["max_seq_length"],
            loss_margin=config["loss_margin"],
        )

        fold_model.eval()
        train_embs = fold_model.encode(
            X_train,
            batch_size=config["batch_size"],
            normalize_embeddings=True,
            convert_to_tensor=True,
        )
        test_embs = fold_model.encode(
            X_test,
            batch_size=config["batch_size"],
            normalize_embeddings=True,
            convert_to_tensor=True,
        )

        centroids = build_centroids(train_embs, y_train)
        pred_centroid, _ = predict_centroid(test_embs, centroids)
        acc_c = accuracy_score(y_test, pred_centroid)
        f1_c = f1_score(y_test, pred_centroid, average="macro")

        pred_knn = predict_knn(train_embs, y_train, test_embs, n_neighbors=config["knn_neighbors"])

        fold_accuracies_centroid.append(acc_c)
        fold_f1_centroid.append(f1_c)
        all_true.extend(y_test)
        all_pred_centroid.extend(pred_centroid)
        all_pred_knn.extend(pred_knn)

        print(f"Fold {fold} Results -> Centroid Acc: {acc_c:.4f}, Macro F1: {f1_c:.4f}")

        del fold_model, train_embs, test_embs
        clear_memory()

    print("\n" + "=" * 80)
    print("5-FOLD CV FINAL RESULTS: NEAREST CENTROID CLASSIFIER")
    print("=" * 80)
    print(f"Mean Accuracy : {np.mean(fold_accuracies_centroid):.4f} +/- {np.std(fold_accuracies_centroid):.4f}")
    print(f"Mean Macro F1 : {np.mean(fold_f1_centroid):.4f} +/- {np.std(fold_f1_centroid):.4f}")
    print("\nClassification Report (Aggregated OOF):")
    print(classification_report(all_true, all_pred_centroid, target_names=class_names, digits=4))

    print("\n" + "=" * 80)
    print("5-FOLD CV FINAL RESULTS: KNN CLASSIFIER")
    print("=" * 80)
    print(f"Overall Accuracy : {accuracy_score(all_true, all_pred_knn):.4f}")
    print(f"Overall Macro F1 : {f1_score(all_true, all_pred_knn, average='macro'):.4f}")
    print("\nClassification Report (Aggregated OOF):")
    print(classification_report(all_true, all_pred_knn, target_names=class_names, digits=4))


# -------------------------- Final Training --------------------------
def train_and_save_final_model(
    df: pd.DataFrame,
    class_names: List[str],
    config: Dict[str, Any] = CONFIG,
) -> None:
    """
    Train contrastive model on the entire dataset and persist model weights, class names, and centroids.
    """
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL ON FULL DATASET")
    print("=" * 80)

    output_dir = config["output_model_dir"]
    texts = df["code_summary"].tolist()
    labels = df["label_enc"].tolist()

    train_embeddings = None
    if config["use_hard_negatives"]:
        print(f"Mining hard negatives using {config['hn_base_model']}...")
        train_embeddings = extract_base_embeddings(
            texts,
            model_name=config["hn_base_model"],
            batch_size=config["batch_size"],
        )

    train_examples = build_contrastive_pairs(
        texts=texts,
        labels=labels,
        embeddings=train_embeddings,
        max_pairs_per_class=config["max_pairs_per_class"],
        num_hard_negatives=config["num_hard_negatives"],
        use_hard_negatives=config["use_hard_negatives"],
    )
    print(f"Total training pairs: {len(train_examples)}")

    final_model = train_contrastive_model(
        model_name=config["model_name"],
        train_examples=train_examples,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        max_seq_length=config["max_seq_length"],
        loss_margin=config["loss_margin"],
    )

    final_model.eval()
    print("Computing class centroids...")
    all_embs = final_model.encode(
        texts,
        batch_size=config["batch_size"],
        normalize_embeddings=True,
        convert_to_tensor=True,
    )
    centroids = build_centroids(all_embs, labels)

    os.makedirs(output_dir, exist_ok=True)
    final_model.save(output_dir)

    with open(os.path.join(output_dir, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    torch.save(centroids, os.path.join(output_dir, "centroids.pt"))
    print(f"Artifacts successfully saved to: {output_dir}")


# -------------------------- Execution --------------------------
if __name__ == "__main__":
    set_seed(CONFIG["seed"])

    print(f"Loading dataset from: {CONFIG['dataset_path']}")
    dataset, label_encoder = load_and_preprocess_data(
        CONFIG["dataset_path"],
        min_samples_per_label=CONFIG["min_samples_per_label"],
    )
    target_class_names = list(label_encoder.classes_)
    print(f"Dataset loaded: {len(dataset)} samples across {len(target_class_names)} classes.")

    # 1. 5-Fold Cross Validation
    run_5fold_cross_validation(dataset, class_names=target_class_names, config=CONFIG)

    # 2. Final Full Dataset Model Training
    train_and_save_final_model(dataset, class_names=target_class_names, config=CONFIG)
