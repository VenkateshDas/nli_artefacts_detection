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

## Experiment Workflow

### Complete Pipeline Flowchart

This diagram shows the entire experimental pipeline from raw data to final results:

```
                    📁 Raw COLIEE Data
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │   STEP 1: Data Preprocessing         │
        │   (data_preprocess.ipynb)            │
        │                                      │
        │   • Tokenize text                    │
        │   • Calculate word overlap           │
        │   • Extract negation words           │
        │   • Detect subsequences              │
        └──────────────┬───────────────────────┘
                       │
                       ▼
              📊 Processed CSV Files
              (with features added)
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
    ┌─────────────┐       ┌─────────────────┐
    │   STEP 2A:  │       │    STEP 2B:     │
    │  Detection  │       │    Training     │
    │             │       │                 │
    │  Analyze    │       │  Train BERT     │
    │  artefacts  │       │  models         │
    └──────┬──────┘       └────────┬────────┘
           │                       │
           ▼                       │
    📈 Statistics &                │
    Adversarial Set                │
           │                       │
           └───────────┬───────────┘
                       ▼
            ┌──────────────────────┐
            │   STEP 3: Evaluate   │
            │                      │
            │  • Normal test set   │
            │  • Adversarial test  │
            │  • Robustness check  │
            └──────────┬───────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
     ✅ Good Results      ❌ Poor Results
     (Robust model)      (Artefact reliance)
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  STEP 4: Augmentation   │
                    │                         │
                    │  Generate balanced data │
                    └────────────┬────────────┘
                                 │
                                 ▼
                          🔄 Retrain Model
                                 │
                                 ▼
                        ✅ Improved Results
```

### How Artefacts Work (Visual Example)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      WORD OVERLAP ARTEFACT                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Premise:  "The contract becomes valid upon signature by both      │
│             parties and must be executed within 30 days."          │
│                                                                     │
│  Hypothesis: "The contract becomes valid upon signature."          │
│                                                                     │
│  Label: YES (Entailment) ✓                                         │
│                                                                     │
│  🔍 Analysis:                                                       │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ Overlap: 7 words match exactly                       │          │
│  │ Overlap ratio: 7/8 = 87.5%                          │          │
│  │                                                      │          │
│  │ ⚠️ Problem: Model learns                             │          │
│  │    "High overlap = YES" instead of                   │          │
│  │    understanding the actual meaning                  │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                     │
│  💡 Adversarial Example (to test robustness):                      │
│  Hypothesis: "The contract expires after signature."               │
│  Overlap: Still high, but meaning contradicts!                     │
│  Expected: NO, Artefact-based model: YES (wrong!) ❌               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                   NEGATION WORD ARTEFACT                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Premise:  "A guarantor has the responsibility to pay debts."      │
│                                                                     │
│  Hypothesis: "A guarantor has no responsibility to pay."           │
│                                                                     │
│  Label: NO (Contradiction) ✓                                       │
│                                                                     │
│  🔍 Analysis:                                                       │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ Negation words found: ["no"]                         │          │
│  │                                                      │          │
│  │ ⚠️ Problem: Model learns                             │          │
│  │    "Has negation = NO" instead of                    │          │
│  │    understanding context                             │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                     │
│  💡 Adversarial Example (to test robustness):                      │
│  Hypothesis: "No other party is liable for the debt."              │
│  Has "no", but doesn't contradict - still entails!                 │
│  Expected: YES, Artefact-based model: NO (wrong!) ❌               │
└─────────────────────────────────────────────────────────────────────┘
```

### Training & Evaluation Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│                    TRAINING EXPERIMENT                         │
└────────────────────────────────────────────────────────────────┘

    📂 Training Data                    ⚙️ Configuration
    ├─ coliee_train_2020.csv          ├─ Model: BERT/RoBERTa/etc.
    ├─ 90% train / 10% validation     ├─ Features: ALL/NONE/specific
    └─ Features extracted              └─ Mode: full-context/hyp-only
              │                                    │
              └────────────┬───────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   🤖 Model Training    │
              │                        │
              │  • 15 epochs           │
              │  • Batch size: 8       │
              │  • Learning rate: 5e-6 │
              │  • Early stopping      │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  💾 Save Best Model    │
              └───────────┬────────────┘
                          │
           ┌──────────────┴──────────────┐
           │                             │
           ▼                             ▼
    ┌─────────────┐              ┌──────────────┐
    │  Test on    │              │   Test on    │
    │  Normal     │              │  Adversarial │
    │  Test Set   │              │   Test Set   │
    └──────┬──────┘              └──────┬───────┘
           │                             │
           ▼                             ▼
    Accuracy: 85%                 Accuracy: 60%
    ✅ Good!                       ⚠️ Artefact reliance!
           │                             │
           └──────────────┬──────────────┘
                          ▼
              ┌────────────────────────┐
              │   📊 Results Analysis  │
              │                        │
              │  • Overall accuracy    │
              │  • Per-artefact acc    │
              │  • For/Against splits  │
              └────────────────────────┘
```

### Data Augmentation Strategy

```
┌──────────────────────────────────────────────────────────────────┐
│              ARTEFACT MITIGATION VIA AUGMENTATION                │
└──────────────────────────────────────────────────────────────────┘

    Original Dataset (Biased)              Augmented Dataset (Balanced)
    ────────────────────────              ──────────────────────────

    High Overlap → YES (80%)              High Overlap → YES (50%)
                                          High Overlap → NO  (50%)
                   ↓                                 ↓
    Model learns shortcut                Model must understand meaning!


    AUGMENTATION PROCESS:
    ─────────────────────

    Step 1: Identify Bias              Step 2: Generate Counter-examples
    ┌──────────────────┐              ┌────────────────────────────┐
    │ Original:        │              │ Augmented:                 │
    │ ───────────      │              │ ─────────────              │
    │ P: "Article 123" │   ───────▶   │ P: "Article 123 ..."       │
    │ H: "Article 123" │   Generate   │ H: "Article 456 ..."       │
    │ Label: YES       │              │ Label: NO                  │
    │ Overlap: 100%    │              │ Overlap: 100% BUT NO!      │
    └──────────────────┘              └────────────────────────────┘

    Step 3: Combine                    Step 4: Retrain
    ┌──────────────────┐              ┌────────────────────────────┐
    │ Original (1000)  │              │ Model now learns:          │
    │      +           │   ────▶      │                            │
    │ Augmented (500)  │   Train      │ "I can't just rely on      │
    │      =           │              │  overlap, I need to read!" │
    │ Total (1500)     │              │                            │
    └──────────────────┘              └────────────────────────────┘
```

### Model Comparison Experiment

```
┌─────────────────────────────────────────────────────────────────────┐
│           COMPARING DIFFERENT EXPERIMENTAL CONDITIONS               │
└─────────────────────────────────────────────────────────────────────┘

Experiment 1: Baseline                Experiment 2: With Features
──────────────────────                ───────────────────────────
Input: Premise + Hypothesis           Input: Premise + Hypothesis + Features
Features: None                        Features: overlap, negations, length

     [BERT Model]                          [BERT Model]
          │                                      │
          ▼                                      ▼
   Normal Test: 85%                       Normal Test: 87%
   Adv Test: 58% ⚠️                       Adv Test: 61% ⚠️


Experiment 3: Hypothesis Only         Experiment 4: Augmented Data
──────────────────────────            ────────────────────────────
Input: Hypothesis only                Input: Premise + Hypothesis
Purpose: Check hyp-only bias          Data: Original + Augmented

     [BERT Model]                          [BERT Model]
          │                                      │
          ▼                                      ▼
   Normal Test: 72%                       Normal Test: 84%
   Adv Test: 45% ⚠️⚠️                     Adv Test: 78% ✅✅
   (High hypothesis bias!)                (Much more robust!)


RESULT INTERPRETATION:
━━━━━━━━━━━━━━━━━━━━━━
Normal Test = How well model works on standard data
Adv Test = How well model resists artefact exploitation
Large gap = Model is cheating with shortcuts! ⚠️
Small gap = Model truly understands! ✅
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

### Quick Reference: What to Run

```
┌─────────────────────────────────────────────────────────────────┐
│                     NOTEBOOK EXECUTION GUIDE                    │
└─────────────────────────────────────────────────────────────────┘

  Goal: Prepare Data
  ├─ Run: src/data scripts/data_preprocess.ipynb
  └─ Output: CSV files with features

  Goal: Understand Dataset Biases
  ├─ Run: src/detection/coliee_artefacts_detection.ipynb
  └─ Output: Statistics + adversarial test set

  Goal: Train & Test Models
  ├─ Run: src/evaluation/Auto_X01_BERT_coliee_models_w_features.ipynb
  ├─ Configure: Model, features, data type
  └─ Output: Trained models + accuracy results

  Goal: Improve Model Robustness
  ├─ Step 1: src/mitigation/create_data_augmentation_instances.ipynb
  ├─ Step 2: src/mitigation/validate_augmented_instances.ipynb
  ├─ Step 3: src/mitigation/combine_augmented_coliee_datasets.ipynb
  └─ Then: Retrain with augmented data

  Goal: Analyze Results
  ├─ Run: src/evaluation/results/model_results_analysis.ipynb
  ├─ Run: src/evaluation/results/Results_Visualization.ipynb
  └─ Output: Plots and comparison tables

  Goal: Interpret Model Decisions
  └─ Run: src/evaluation/model interpretability/BERT_Model_Interpretability.ipynb
```

---

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

**Which configuration should I use?** Follow this decision tree:

```
                    What do you want to test?
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
    Baseline model      Check if model      Test robustness
    performance         uses shortcuts      after mitigation
         │                    │                    │
         ▼                    ▼                    ▼
    DATA_TYPE =         MODEL_TYPE =          DATA_TYPE =
    "Normal"            "hyp-only"            "Aug"
    MODEL_TYPE =        (tests if model       MODEL_TYPE =
    "full-context"      can work without      "full-context"
    feature_name =      premise - bad         feature_name =
    ["NONE"]            sign!)                ["NONE"]

         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
    Compare models      Test feature        Legal domain
    (BERT vs RoBERTa)   importance          specific
         │                    │                    │
         ▼                    ▼                    ▼
    chosen_model =      feature_name =        chosen_model =
    [loop through       ["ALL"] vs            models['LEGAL_BERT']
    all models]         ["NONE"] vs
                        individual features


    💡 RECOMMENDED EXPERIMENT SEQUENCE:
    ──────────────────────────────────
    1. Baseline: Normal data, no features, full-context
       → See how model performs naturally

    2. Hypothesis-only: Normal data, hyp-only mode
       → Check if hypothesis alone gives high accuracy (BAD!)

    3. Feature ablation: Normal data, test each feature
       → Understand which artefacts model relies on

    4. Adversarial test: Use adversarial test set
       → Quantify how much model cheats

    5. Augmented training: Aug data, no features
       → Train on balanced data

    6. Compare results: Normal vs Aug performance
       → Measure improvement in robustness
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

### Understanding Your Results

```
┌─────────────────────────────────────────────────────────────────┐
│                   RESULT INTERPRETATION GUIDE                   │
└─────────────────────────────────────────────────────────────────┘

EXAMPLE OUTPUT:
──────────────

Normal Test Accuracy: 85%
Adversarial Test Accuracy: 60%
  ├─ Against (contradicting artefact): 45%
  └─ For (exploiting artefact): 75%

Artefact-specific accuracy:
  ├─ Word Overlap: 55%
  └─ Contradiction Words: 65%


WHAT DOES THIS MEAN?
─────────────────────

✅ GOOD SIGNS:
━━━━━━━━━━━━━
1. Small gap between Normal and Adversarial accuracy
   Example: Normal 85%, Adv 80% → Only 5% drop ✓

2. Similar accuracy for "Against" and "For" examples
   Example: Against 78%, For 82% → Balanced ✓

3. High accuracy on all artefact types
   Example: All artefacts > 75% → Robust ✓


⚠️ WARNING SIGNS:
━━━━━━━━━━━━━━━
1. Large gap between Normal and Adversarial accuracy
   Example: Normal 85%, Adv 55% → 30% drop! Model relies on artefacts

2. Big difference between "Against" and "For"
   Example: Against 45%, For 75% → 30% gap! Model exploits shortcuts

3. Low accuracy on specific artefact type
   Example: Word Overlap 40% → Model cheats using overlap


TYPICAL RESULTS BY MODEL TYPE:
───────────────────────────────

┌─────────────────────┬──────────┬──────────┬──────────────┐
│ Model Configuration │ Normal   │ Adv      │ Robustness   │
├─────────────────────┼──────────┼──────────┼──────────────┤
│ Baseline (Normal)   │   85%    │   58%    │ ⚠️ Poor      │
│ With Features       │   87%    │   62%    │ ⚠️ Slight    │
│ Hyp-only            │   72%    │   45%    │ ⚠️⚠️ Very Bad │
│ Augmented Data      │   84%    │   78%    │ ✅ Good!     │
│ Legal-BERT + Aug    │   88%    │   82%    │ ✅✅ Excellent│
└─────────────────────┴──────────┴──────────┴──────────────┘


HOW TO IMPROVE POOR RESULTS:
─────────────────────────────

Problem: Large accuracy drop on adversarial test
Solution: ① Use data augmentation
          ② Train on multiple years combined
          ③ Try domain-specific model (Legal-BERT)

Problem: Hypothesis-only achieves high accuracy
Solution: ① Clear hypothesis bias in dataset
          ② Must use augmentation
          ③ Consider creating more diverse data

Problem: Low accuracy on specific artefact
Solution: ① Generate more counter-examples for that artefact
          ② Analyze what patterns model learns
          ③ Use model interpretability notebook


READING THE CSV PREDICTIONS:
─────────────────────────────

{run_name}-adversarial_instance_predictions.csv contains:

id, label, premise, hypothesis, Artefact Type, Adv Type, predictions
│    │      │        │           │              │          │
│    │      │        │           │              │          └─ Model's prediction
│    │      │        │           │              └─ For/Against
│    │      │        │           └─ Which artefact is tested
│    │      │        └─ Hypothesis text
│    │      └─ Premise text
│    └─ Ground truth label (0=No, 1=Yes)
└─ Instance ID

Look for rows where: label != predictions
→ These are errors! Analyze them to understand model weaknesses.
```

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