# Source Code Execution Guide

This directory contains the end-to-end implementation for mining, detecting, and classifying AI design patterns in software repositories, as described in the paper *"A Methodology for Investigating AI Patterns Prevalence in Software Repositories"*.

---

## 1. Prerequisites & Environment Setup

### Hardware Requirements
- **Standard Pipeline (Steps 1, 2, 3 ML, 4)**: Standard multi-core CPU / standard GPU (Gemini API handles embedding & LLM workloads).
- **Supervised Contrastive Learning (`step-03`)**: Requires **> 50 GB GPU VRAM** due to fine-tuning the 8B-parameter transformer backbone (`BAAI/bge-reasoner-embed-qwen3-8b-0923`). An **NVIDIA A100 (80GB)** or equivalent high-memory accelerator is strongly recommended.

### Environment Variables
Set your Google Gemini API key:
```bash
export GOOGLE_API_KEY="your-google-api-key"
# or on Windows PowerShell:
# $env:GOOGLE_API_KEY="your-google-api-key"
```

### Key Python Dependencies
Install required packages:
```bash
pip install numpy pandas scikit-learn networkx python-louvain \
            langchain langchain-google-genai google-generativeai \
            sentence-transformers torch umap-learn pypdf joblib scipy
```

---

## 2. Standard Pipeline Workflow

### Step 1: Pattern Mining & Clustering
Extract candidate design patterns from research paper PDFs and cluster them.

1. **Extract Patterns from Literature**:
   ```bash
   python step-01/step1_extract_patterns.py
   ```
   - *Input*: Research paper PDFs placed in `data/raw/papers/<tag>/`.
   - *Output*: Extracted pattern JSON files under `outputs/<tag>/<run>/`.

2. **Cluster & Summarize Extracted Patterns**:
   ```bash
   python step-01/step1_cluster_and_summarize_patterns.py
   ```
   - *Input*: `outputs/<tag>/<run>/extracted_patterns/l2_patterns_v2.json`.
   - *Process*: Generates Gemini embeddings (`gemini-embedding-001`), performs UMAP dimension reduction, and clusters patterns using DBSCAN.

---

### Step 2: Codebase Parsing & Community Embeddings
Construct function-level call graphs from target repositories, detect functional communities, and embed them.

1. **Build Call Graphs & Detect Communities**:
   ```bash
   python step-02/step2_callgraph_and_communities.py
   ```
   - *Input*: Repositories placed in `repos/cloned_repos/`.
   - *Process*: Parses Python ASTs, builds directed call graphs, and applies Louvain community detection.
   - *Output*: Clustered community `.py` files under `result/repo_callgraph_clusters/`.

2. **Generate Code Embeddings for Communities**:
   ```bash
   python step-02/step2_repo_code_to_embeddings.py
   ```
   - *Process*: Generates Gemini embeddings (`text-embedding-004`) for all community code files.
   - *Output*: `results/pattern_embeddings/callgraph_embeddings.csv`.

---

### Step 3: Dataset Preparation & Model Training (Ensemble Classifiers)

1. **Generate Synthetic Bootstrap Samples (Optional)**:
   ```bash
   python step-03/step3_generate_bootstrap_samples.py
   ```
   - *Process*: Uses an LLM agent to synthesize code summaries for patterns with limited real-world instances.

2. **Generate Embeddings for Training Data**:
   ```bash
   python step-03/step3_training_code_to_embeddings.py
   ```
   - *Process*: Computes `text-embedding-004` vectors for all labeled and bootstrap samples.

3. **Train & Evaluate Classifiers (5-Fold CV)**:
   ```bash
   python step-03/step3_train_models_cv.py
   ```
   - *Process*: Evaluates Logistic Regression, SVC, and KNN across 5-fold Stratified Cross-Validation, builds voting ensembles, and saves trained models into `artifacts/models/`.

---

### Step 4: Forecasting, Inference & Statistical Correction

1. **Predict Pattern Labels for Unverified Code**:
   ```bash
   python step-04/step4_forecast_and_simulate.py
   ```
   - *Process*: Runs the weighted voting ensemble on unverified code communities and saves predictions to `artifacts/outputs/unverified_weighted_predictions.csv`.

2. **Simulate & Correct Observed Counts via Confusion Matrix**:
   ```bash
   python step-04/simulate_counts_confusion_matrix.py
   ```
   - *Process*: Uses constrained linear optimization (SLSQP via `scipy.optimize.minimize`) to adjust observed pattern frequencies based on classifier error matrices and Monte Carlo simulations.

---

## 3. Standalone Module: Supervised Contrastive Learning

As an alternative to standard classifier ensembles, a dedicated supervised contrastive learning pipeline is provided in `step-03`. It fine-tunes transformer embeddings using **hard negative mining** and **contrastive loss**, classifying samples via **nearest class centroids**.

> **Hardware Requirement:** Fine-tuning the 8B-parameter model (`BAAI/bge-reasoner-embed-qwen3-8b-0923`) requires **> 50 GB GPU VRAM**. We strongly recommend using an **NVIDIA A100 (80GB)** GPU.

### 1. Training & Cross-Validation
```bash
python step-03/step3_contrastive_learning_train.py
```
- **What it does**:
  1. Loads labeled dataset and extracts embeddings using a lightweight base model (`google-bert/bert-base-uncased`) to mine hard negative pairs.
  2. Constructs positive (intra-class) and hard negative (inter-class) pairs.
  3. Evaluates out-of-fold generalization with **5-fold Stratified Group Cross-Validation**.
  4. Trains the final `SentenceTransformer` model on the full dataset with `ContrastiveLoss`.
  5. Computes L2-normalized class centroid vectors.
- **Output Artifacts** (saved to `./saved_models/final_model/`):
  - Model weights & tokenizer files
  - `class_names.json`: Target class name mapping
  - `centroids.pt`: Precomputed class centroid embeddings

### 2. Inference & Pattern Prediction
Classify new code summaries or batch CSV files:

- **Single text query**:
  ```bash
  python step-03/step3_contrastive_learning_predict.py --text "Implements a circuit breaker with timeout and fallback routing."
  ```

- **Batch CSV classification**:
  ```bash
  python step-03/step3_contrastive_learning_predict.py --input_csv path/to/input.csv --output_csv path/to/output.csv --text_col code_summary
  ```

- **Run built-in demo**:
  ```bash
  python step-03/step3_contrastive_learning_predict.py
  ```
