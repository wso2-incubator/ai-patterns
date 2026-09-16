"""Inference pipeline: load fine-tuned contrastive model and class centroids to score code patterns."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F


# -------------------------- Config --------------------------
DEFAULT_MODEL_DIR = "./saved_models/final_model"


# -------------------------- Model Loading --------------------------
def load_inference_artifacts(
    model_dir: str = DEFAULT_MODEL_DIR,
) -> Tuple[SentenceTransformer, List[str], Dict[int, torch.Tensor]]:
    """
    Load fine-tuned SentenceTransformer model, class name list, and precomputed class centroids.
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model directory not found at: '{model_dir}'. "
            f"Please run 'step3_contrastive_learning_train.py' first."
        )

    class_names_path = os.path.join(model_dir, "class_names.json")
    centroids_path = os.path.join(model_dir, "centroids.pt")

    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Missing '{class_names_path}'.")
    if not os.path.exists(centroids_path):
        raise FileNotFoundError(f"Missing '{centroids_path}'.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from '{model_dir}' onto {device}...")
    model = SentenceTransformer(model_dir, device=device)

    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    centroids = torch.load(centroids_path, map_location="cpu")
    # Normalize centroid keys to integer and ensure tensors are normalized
    normalized_centroids = {}
    for k, v in centroids.items():
        normalized_centroids[int(k)] = F.normalize(v.float().cpu(), dim=-1)

    print(f"Successfully loaded {len(class_names)} target classes and centroids.")
    return model, class_names, normalized_centroids


# -------------------------- Inference --------------------------
def predict(
    model: SentenceTransformer,
    centroids: Dict[int, torch.Tensor],
    class_names: List[str],
    texts: List[str],
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    Perform nearest-centroid inference on a list of text strings and return top predictions with confidence scores.
    """
    model.eval()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=len(texts) > 50,
    )

    classes = sorted(centroids.keys())
    centroid_mat = torch.stack([centroids[c] for c in classes], dim=0).to(embeddings.device).float()

    # Cosine similarity matrix: shape (num_samples, num_classes)
    sims = (embeddings.float() @ centroid_mat.T).cpu().numpy()

    results = []
    for idx, text in enumerate(texts):
        sample_sims = sims[idx]
        sorted_indices = np.argsort(sample_sims)[::-1]

        top1_idx = sorted_indices[0]
        top1_class = class_names[classes[top1_idx]]
        top1_score = float(sample_sims[top1_idx])

        top2_idx = sorted_indices[1] if len(classes) > 1 else top1_idx
        top2_class = class_names[classes[top2_idx]]
        top2_score = float(sample_sims[top2_idx])

        results.append({
            "text": text,
            "predicted_class": top1_class,
            "confidence_score": top1_score,
            "top2_predicted_class": top2_class,
            "top2_confidence_score": top2_score,
        })

    return results


def predict_csv(
    model: SentenceTransformer,
    centroids: Dict[int, torch.Tensor],
    class_names: List[str],
    input_csv_path: str,
    output_csv_path: str,
    text_column: str = "code_summary",
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Classify texts from an input CSV file and save predictions with confidence scores to an output CSV.
    """
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")

    df = pd.read_csv(input_csv_path)
    if text_column not in df.columns:
        if "code" in df.columns:
            text_column = "code"
        else:
            raise ValueError(f"Column '{text_column}' not found in CSV. Columns are: {list(df.columns)}")

    texts = df[text_column].fillna("").astype(str).tolist()
    print(f"Running inference on {len(texts)} samples from '{input_csv_path}'...")
    predictions = predict(model, centroids, class_names, texts, batch_size=batch_size)

    pred_df = pd.DataFrame(predictions)
    result_df = df.copy()
    result_df["predicted_label"] = pred_df["predicted_class"]
    result_df["confidence_score"] = pred_df["confidence_score"]
    result_df["top2_predicted_label"] = pred_df["top2_predicted_class"]
    result_df["top2_confidence_score"] = pred_df["top2_confidence_score"]

    result_df.to_csv(output_csv_path, index=False)
    print(f"Saved predictions to '{output_csv_path}'.")
    return result_df


# -------------------------- Main / CLI --------------------------
def main() -> None:
    """CLI entrypoint for single sample, batch CSV, or demo query prediction."""
    parser = argparse.ArgumentParser(description="Predict code pattern classes using trained centroid model.")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Path to saved model folder")
    parser.add_argument("--text", type=str, default=None, help="Single code summary text to classify")
    parser.add_argument("--input_csv", type=str, default=None, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, default="predictions_output.csv", help="Path to save output CSV")
    parser.add_argument("--text_col", type=str, default="code_summary", help="Text column name in input CSV")
    args = parser.parse_args()

    # Load model and centroids
    model, class_names, centroids = load_inference_artifacts(args.model_dir)

    # 1. Single sample prediction
    if args.text:
        res = predict(model, centroids, class_names, [args.text])[0]
        print("\n" + "=" * 70)
        print("PREDICTION RESULT")
        print("=" * 70)
        print(f"Query Text  : {res['text']}")
        print(f"Predicted   : {res['predicted_class']} (Confidence: {res['confidence_score']:.4f})")
        print(f"Second Best : {res['top2_predicted_class']} (Confidence: {res['top2_confidence_score']:.4f})")
        return

    # 2. Batch CSV prediction
    if args.input_csv:
        predict_csv(
            model=model,
            centroids=centroids,
            class_names=class_names,
            input_csv_path=args.input_csv,
            output_csv_path=args.output_csv,
            text_column=args.text_col,
        )
        return

    # 3. Default demo queries if no arguments provided
    demo_queries = [
        "Implements token-based user authentication and claims verification middleware for securing endpoints.",
        "Implements a circuit breaker pattern with failure rate threshold, timeout limits, and fallback service routing.",
        "Publishes event messages to a partitioned Kafka topic with guaranteed delivery and idempotent retries.",
    ]
    print("\nRunning demonstration with built-in test queries:")
    results = predict(model, centroids, class_names, demo_queries)
    print("\n" + "=" * 80)
    print("DEMO INFERENCE RESULTS")
    print("=" * 80)
    for r in results:
        print(f"Query       : {r['text']}")
        print(f"Prediction  : {r['predicted_class']} (Confidence: {r['confidence_score']:.4f})")
        print(f"Runner-up   : {r['top2_predicted_class']} (Confidence: {r['top2_confidence_score']:.4f})\n")


if __name__ == "__main__":
    main()
