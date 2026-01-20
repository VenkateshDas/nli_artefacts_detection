# Detecting and Mitigating Language Artefacts in Legal NLI Datasets

<div align="center">

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-research-orange.svg)

*A comprehensive framework for detecting, evaluating, and mitigating dataset artefacts in Natural Language Inference tasks for legal text*

</div>

---

## Table of Contents

- [Overview](#overview)
- [The Problem](#the-problem)
- [Our Approach](#our-approach)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Setup](#data-setup)
- [Usage](#usage)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Artefact Detection](#2-artefact-detection)
  - [3. Model Training & Evaluation](#3-model-training--evaluation)
  - [4. Data Augmentation](#4-data-augmentation)
- [Supported Models](#supported-models)
- [Research Methodology](#research-methodology)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

This repository contains the complete codebase for a research project on **detecting, evaluating, and mitigating language artefacts** in the COLIEE (Competition on Legal Information Extraction and Entailment) dataset.

The project is part of a thesis investigating how unintended statistical patterns (artefacts) in training data can allow NLI models to "cheat" by exploiting shortcuts rather than learning genuine semantic understanding.

> **Note:** This repository is a work in progress. Scripts are being actively refined and improved.

---

## The Problem

Natural Language Inference (NLI) models, particularly BERT-based architectures, can achieve high accuracy on benchmark datasets by exploiting superficial patterns rather than truly understanding the relationship between premises and hypotheses. This is especially problematic in legal NLI tasks where reasoning accuracy is critical.

### Common Dataset Artefacts

| Artefact Type | Description | Example |
|---------------|-------------|---------|
| **Word Overlap** | High word overlap often indicates entailment | Premise and hypothesis share 80%+ of words → likely "Yes" |
| **Contradiction Words** | Presence of negation words signals contradiction | Words like "not", "never", "no" → likely "No" |
| **Subsequence Heuristic** | Hypothesis appears verbatim in premise | Exact match of hypothesis text → likely "Yes" |
| **Annotation Artefacts** | Dataset-specific patterns from labeling process | Mentions of "building" in legal context → biased labels |

---

## Our Approach

This project implements a three-phase methodology:

```
┌─────────────────────┐
│   1. DETECTION      │  Analyze datasets to identify and quantify artefacts
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   2. EVALUATION     │  Test model robustness with adversarial examples
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   3. MITIGATION     │  Apply data augmentation to reduce artefact reliance
└─────────────────────┘
```

---

## Features

✅ **Comprehensive Artefact Detection**
- Word overlap analysis with configurable thresholds
- Negation word extraction and frequency analysis
- Subsequence pattern matching
- Statistical bias quantification

✅ **Multi-Model Evaluation**
- Support for 7+ pre-trained transformer models
- Both full-context and hypothesis-only evaluation modes
- Feature-based vs. feature-free comparisons
- Adversarial test set generation

✅ **Data Augmentation Pipeline**
- Automated generation of balanced training instances
- Validation framework for augmented data quality
- Combination utilities for merging datasets

✅ **Experiment Tracking**
- Weights & Biases integration
- Detailed logging and metrics
- Model checkpoint management
- Result visualization

---

## Project Structure

```
nli_artefacts_detection/
│
├── src/
│   ├── data scripts/           # Data processing pipeline
│   │   ├── data_parse.ipynb           # Parse raw COLIEE data
│   │   ├── data_preprocess.ipynb      # Feature engineering
│   │   └── data_analysis.ipynb        # Exploratory analysis
│   │
│   ├── detection/              # Artefact detection
│   │   ├── coliee_artefacts_detection.ipynb
│   │   └── misc/                      # Additional experiments (MNLI, SNLI)
│   │
│   ├── evaluation/             # Model training & testing
│   │   ├── Auto_X01_BERT_coliee_models_w_features.ipynb
│   │   ├── Auto_X02_Custom_BERT_models_w_features_nli.ipynb
│   │   ├── adversarial test set/      # Adversarial evaluation
│   │   ├── model interpretability/    # Model explanations
│   │   └── results/                   # Analysis & visualization
│   │
│   ├── mitigation/             # Data augmentation
│   │   ├── create_data_augmentation_instances.ipynb
│   │   ├── validate_augmented_instances.ipynb
│   │   └── combine_augmented_coliee_datasets.ipynb
│   │
│   └── misc/                   # Experimental models (BiLSTM, etc.)
│
├── data/                       # Dataset storage (not in repo)
├── models/                     # Trained model checkpoints (not in repo)
├── plots/                      # Generated visualizations (not in repo)
│
├── CLAUDE.md                   # AI assistant guide
├── README.md                   # This file
├── LICENSE                     # MIT License
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM
- COLIEE dataset access (register at COLIEE competition website)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/VenkateshDas/nli_artefacts_detection.git
   cd nli_artefacts_detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers datasets evaluate
   pip install pandas numpy nltk scikit-learn
   pip install jupyter notebook
   pip install wandb  # For experiment tracking (optional)
   ```

4. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('punkt')"
   ```

### Data Setup

1. **Create data directory structure**
   ```bash
   mkdir -p data/coliee_data/processed/task4/{train,test}
   mkdir -p data/coliee_data/task4/adv-task4/test
   ```

2. **Obtain COLIEE dataset**
   - Register at the [COLIEE competition website](http://www.coliee.org/)
   - Download Task 4 (Legal Case Entailment) data
   - Place raw files in `data/coliee_data/`

3. **Verify setup**
   ```bash
   ls data/coliee_data/  # Should show your downloaded COLIEE files
   ```

---

## Usage

### 1. Data Preprocessing

First, preprocess the raw COLIEE data to extract features and create analysis-ready CSV files.

**Run:** `src/data scripts/data_preprocess.ipynb`

This notebook will:
- Parse premise-hypothesis pairs
- Tokenize text using NLTK
- Calculate word overlap metrics
- Extract negation words
- Detect subsequence patterns
- Generate feature columns

**Output:** CSV files with columns:
```
id, label, premise, hypothesis, labels, hyp_tokens, hyp_length,
prem_tokens, prem_length, overlap, is_word_overlap, negations,
has_negation, is_subsequence_heuristic
```

### 2. Artefact Detection

Analyze the preprocessed data to identify and quantify artefacts.

**Run:** `src/detection/coliee_artefacts_detection.ipynb`

This notebook will:
- Generate statistics on word overlap distribution
- Analyze negation word frequency by label
- Identify subsequence patterns
- Create visualizations of artefact prevalence
- Generate adversarial test examples

**Output:**
- Statistical reports
- Adversarial test set CSV
- Artefact distribution plots

### 3. Model Training & Evaluation

Train transformer models and evaluate their robustness to artefacts.

**Run:** `src/evaluation/Auto_X01_BERT_coliee_models_w_features.ipynb`

**Configuration options:**

```python
# Choose your settings
SEED = 42
DATA_TYPE = "Aug"  # "Normal" or "Aug" (augmented)
MODEL_TYPE = "full-context"  # "full-context" or "hyp-only"
years = ["2018", "2019", "2020", "2021", "2022"]
feature_name = ["NONE", "ALL", "SENTENCE_LENGTH", "WORD_OVERLAP",
                "HAS_CONTRADICTION_WORDS", "SUBSEQUENCE_HEURISTICS"]
chosen_model = models['BERT_BASE']  # See Supported Models section
```

**The notebook will:**
- Load and prepare datasets
- Initialize the selected model
- Train with early stopping
- Evaluate on normal test set
- Evaluate on adversarial test set
- Log results to Weights & Biases (optional)
- Save predictions and metrics

**Output:**
- Trained model checkpoints in `models/`
- Prediction CSVs with instance-level results
- Accuracy metrics (overall, by artefact type, by direction)
- Training logs

### 4. Data Augmentation

Mitigate artefacts by generating balanced training data.

**Step 1:** Create augmentations
```
Run: src/mitigation/create_data_augmentation_instances.ipynb
```

**Step 2:** Validate quality
```
Run: src/mitigation/validate_augmented_instances.ipynb
```

**Step 3:** Combine datasets
```
Run: src/mitigation/combine_augmented_coliee_datasets.ipynb
```

**Output:** Augmented training files (e.g., `coliee_aug_train_2020.csv`)

---

## Supported Models

The framework supports the following pre-trained models:

| Model | Identifier | Specialization |
|-------|-----------|----------------|
| BERT Base | `bert-base-uncased` | General NLI |
| BERT Base MNLI | `gchhablani/bert-base-cased-finetuned-mnli` | MNLI fine-tuned |
| RoBERTa Base | `roberta-base` | General NLI |
| RoBERTa MNLI | `textattack/roberta-base-MNLI` | MNLI fine-tuned |
| Legal-BERT | `nlpaueb/legal-bert-base-uncased` | Legal domain |
| ELECTRA MNLI | `howey/electra-base-mnli` | MNLI fine-tuned |
| DeBERTa NLI | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | Multi-task NLI |

**To add a new model:**
1. Add to `models` dictionary in the evaluation notebook
2. Create corresponding config class
3. Update `get_model_config()` function

---

## Research Methodology

### Phase 1: Detection

1. **Statistical Analysis**: Compute artefact metrics across entire dataset
2. **Correlation Studies**: Measure correlation between artefacts and labels
3. **Visualization**: Generate distribution plots and heatmaps

### Phase 2: Evaluation

1. **Baseline Training**: Train models without artefact-related features
2. **Feature Ablation**: Test models with individual feature sets
3. **Adversarial Testing**: Evaluate on carefully crafted adversarial examples
4. **Robustness Metrics**: Calculate accuracy drops on adversarial data

### Phase 3: Mitigation

1. **Augmentation Strategy**: Generate counter-examples to balance artefacts
2. **Quality Control**: Validate augmented instances for semantic correctness
3. **Retraining**: Train models on augmented datasets
4. **Comparison**: Compare robustness before and after mitigation

---

## Results

Results are saved in multiple formats:

### Text Files
Located in `data/` (if using the notebook paths):
- `Normal-full_context_test_accuracy_{seed}.txt` - Normal test accuracy
- `Normal-FullContextModels_adversarial_test_accuracy_{seed}.txt` - Adversarial accuracy

### CSV Files
Located in `models/{run_name}/`:
- `{run_name}-instance_predictions.csv` - Instance-level predictions
- `{run_name}-adversarial_instance_predictions.csv` - Adversarial predictions

### Visualizations
Generated in `src/evaluation/results/`:
- Model comparison plots
- Artefact distribution charts
- Robustness analysis graphs

### Weights & Biases
If enabled, view interactive dashboards at: https://wandb.ai

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{murugadas2023artefacts,
  author = {Murugadas, Venkatesh},
  title = {Detection, Evaluation and Mitigation of Language Artefacts in Legal NLI Datasets},
  year = {2023},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/VenkateshDas/nli_artefacts_detection}}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2023 Venkatesh Murugadas

---

## Contact

**Author:** Venkatesh Murugadas

**Issues:** Please report bugs or feature requests through [GitHub Issues](https://github.com/VenkateshDas/nli_artefacts_detection/issues)

**Contributing:** Contributions are welcome! Please feel free to submit a Pull Request.

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for the NLP research community

</div>