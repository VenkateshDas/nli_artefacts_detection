# CLAUDE.md - AI Assistant Guide for NLI Artefacts Detection

This document provides comprehensive guidance for AI assistants working with this codebase. Last updated: 2026-01-20

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Technology Stack](#technology-stack)
4. [Development Workflow](#development-workflow)
5. [Code Conventions](#code-conventions)
6. [Key Components](#key-components)
7. [Data Pipeline](#data-pipeline)
8. [Important Notes](#important-notes)
9. [Common Tasks](#common-tasks)

---

## Project Overview

### Purpose
This repository contains the codebase for a thesis work on **detecting, evaluating, and mitigating language/dataset artefacts in legal information entailment datasets**, specifically the COLIEE (Competition on Legal Information Extraction and Entailment) dataset.

### Research Focus
The project investigates three main areas:
1. **Detection**: Identifying language artefacts in NLI datasets
2. **Evaluation**: Testing BERT-based models for robustness against artefacts
3. **Mitigation**: Using data augmentation to reduce the impact of artefacts

### Key Artefacts Studied
- **Word Overlap Bias**: High word overlap between premise and hypothesis
- **Contradiction Words**: Presence of negation words (not, no, n't, none, neither, never, etc.)
- **Subsequence Heuristics**: Hypothesis appearing as subsequence in premise
- **Annotation Artefacts**: Dataset-specific patterns (e.g., Y-Building, Y-Person, N-Rescind, N-Property)

### Status
⚠️ **WIP (Work in Progress)** - The scripts are not yet finalized.

---

## Repository Structure

```
nli_artefacts_detection/
├── .git/                           # Git version control
├── .gitignore                      # Ignored files (data/, models/, plots/, etc.)
├── LICENSE                         # MIT License (Copyright 2023 Venkatesh Murugadas)
├── README.md                       # Basic project documentation
├── CLAUDE.md                       # This file - AI assistant guide
│
└── src/                            # Source code directory
    ├── data scripts/               # Data analysis and preprocessing
    │   ├── data_analysis.ipynb     # Exploratory data analysis
    │   ├── data_parse.ipynb        # Parsing raw data files
    │   └── data_preprocess.ipynb   # Feature engineering and preprocessing
    │
    ├── detection/                  # Artefact detection scripts
    │   ├── coliee_artefacts_detection.ipynb  # Main detection notebook for COLIEE
    │   └── misc/                   # Additional detection experiments
    │       ├── mnli_artefacts_detection.ipynb
    │       └── snli_artefacts_detection.ipynb
    │
    ├── evaluation/                 # Model evaluation and robustness testing
    │   ├── Auto_X01_BERT_coliee_models_w_features.ipynb  # Main training script
    │   ├── Auto_X02_Custom_BERT_models_w_features_nli.ipynb
    │   ├── adversarial test set/   # Adversarial evaluation scripts
    │   │   └── Auto_Adversarial_Test_inference.ipynb
    │   ├── model interpretability/ # Model explanation and analysis
    │   │   └── BERT_Model_Interpretability.ipynb
    │   └── results/                # Results visualization and analysis
    │       ├── model_results_analysis.ipynb
    │       └── Results_Visualization.ipynb
    │
    ├── mitigation/                 # Data augmentation for mitigation
    │   ├── create_data_augmentation_instances.ipynb
    │   ├── validate_augmented_instances.ipynb
    │   └── combine_augmented_coliee_datasets.ipynb
    │
    └── misc/                       # Miscellaneous experiments
        └── BiLSTM_NLI_models.ipynb
```

### Ignored Directories (Not in Git)
These directories are created during execution but are gitignored:
- `data/` - Dataset storage (placeholder, must be populated manually)
- `models/` - Trained model checkpoints
- `plots/` - Visualization outputs
- `.vector_cache/` - Cached embeddings

---

## Technology Stack

### Core Dependencies
Based on the notebooks analyzed, the following libraries are used:

#### Data Processing
- **pandas**: DataFrame manipulation and CSV operations
- **numpy**: Numerical operations
- **nltk**: Natural language processing and tokenization
- **datasets** (HuggingFace): Dataset loading and manipulation

#### Machine Learning
- **torch** (PyTorch): Deep learning framework
- **transformers** (HuggingFace): Pre-trained transformer models
- **evaluate**: Model evaluation metrics
- **sentencepiece**: Tokenization for certain models

#### Training Infrastructure
- **wandb**: Experiment tracking and logging
- **ray[tune]**: Hyperparameter optimization (optional)

#### Utilities
- **tqdm**: Progress bars
- **warnings**: Suppress warnings in notebooks
- **logging**: Control logging output

### Model Architectures Used
The codebase supports multiple pre-trained models:
- **BERT Base**: `bert-base-uncased`
- **BERT Base MNLI**: `gchhablani/bert-base-cased-finetuned-mnli`
- **RoBERTa Base**: `roberta-base`
- **RoBERTa Base MNLI**: `textattack/roberta-base-MNLI`
- **Legal-BERT**: `nlpaueb/legal-bert-base-uncased`
- **ELECTRA Base MNLI**: `howey/electra-base-mnli`
- **DeBERTa Base NLI**: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`

### Development Environment
- **Python**: 3.10+ (based on `.gitignore` patterns)
- **Jupyter Notebooks**: Primary development interface
- **Google Colab**: Some notebooks are designed for Colab (with drive mounting and gcsfuse)
- **CUDA**: GPU support for PyTorch training

---

## Development Workflow

### Setting Up Environment

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd nli_artefacts_detection
   ```

2. **Install Dependencies**
   ```bash
   pip install pandas numpy nltk torch transformers datasets evaluate wandb sentencepiece tqdm
   ```

3. **Download NLTK Data**
   ```python
   import nltk
   nltk.download('punkt')
   ```

4. **Create Data Directory**
   ```bash
   mkdir -p data/coliee_data/processed/task4/{train,test}
   ```

5. **Configure Weights & Biases** (if using experiment tracking)
   ```bash
   wandb login
   ```

### Typical Research Workflow

1. **Data Preprocessing** (`src/data scripts/`)
   - Parse raw COLIEE data files
   - Add features (word overlap, negations, sentence length, etc.)
   - Save processed CSV files

2. **Artefact Detection** (`src/detection/`)
   - Analyze dataset for bias patterns
   - Generate statistics on artefacts
   - Create adversarial test sets

3. **Model Training** (`src/evaluation/`)
   - Train BERT-based models with/without features
   - Experiment with full-context vs hypothesis-only models
   - Track experiments with W&B

4. **Evaluation** (`src/evaluation/`)
   - Test on normal test sets
   - Test on adversarial test sets
   - Analyze robustness to specific artefacts

5. **Mitigation** (`src/mitigation/`)
   - Generate augmented data instances
   - Validate augmentations
   - Combine augmented datasets
   - Retrain and evaluate

6. **Analysis** (`src/evaluation/results/`)
   - Visualize results
   - Compare model performance
   - Generate plots and tables

---

## Code Conventions

### Data Format
All datasets use CSV format with the following core columns:
- `id`: Unique instance identifier
- `label`: Binary label (0 or 1)
- `premise`: The premise text
- `hypothesis`: The hypothesis text
- `labels`: String label ('Y' or 'N')

### Feature Engineering Columns
Preprocessed datasets include additional columns:
- `hyp_tokens`: Tokenized hypothesis (list)
- `hyp_length`: Length of hypothesis in tokens (int)
- `prem_tokens`: Tokenized premise (list)
- `prem_length`: Length of premise in tokens (int)
- `overlap`: Word overlap count (int)
- `is_word_overlap`: Boolean flag for overlap existence
- `negations`: List of negation words found
- `has_negation`: Boolean flag for negation presence
- `is_subsequence_heuristic`: Boolean flag for subsequence detection

### Adversarial Test Set Columns
Additional columns for adversarial evaluation:
- `Artefact Type`: Type of artefact (e.g., "Word Overlap", "Contradiction Word", "Annotation Artefact")
- `Adv Type`: Direction ("For" or "Against" the artefact)

### Naming Conventions

#### File Naming
- Training files: `coliee_train_{year}.csv` or `coliee_aug_train_{year}.csv`
- Test files: `coliee_test_{year}.csv`
- Adversarial test: `adversarial_test_set.csv`

#### Model Run Naming
Format: `X01-{DATA_TYPE}-run-{year}-{run_num}-{MODEL_TYPE}-{features}-features`
- `DATA_TYPE`: "Normal" or "Aug" (augmented)
- `MODEL_TYPE`: "full-context" or "hyp-only"
- `features`: Feature set used (e.g., "ALL", "NONE", "WORD_OVERLAP")

#### Variables
- **Constants**: ALL_CAPS (e.g., `SEED`, `MODEL_TYPE`)
- **Functions**: snake_case (e.g., `calculate_overlap`, `detect_word_overlap_bias`)
- **Classes**: PascalCase (e.g., `CustomDataset`, `GlobalConfig`)
- **Dataclasses**: Used extensively for configuration management

### Python Style

#### Imports Organization
```python
# Standard library
import os
import random
from typing import Tuple

# Third-party
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer

# Suppress warnings (common in notebooks)
import warnings
warnings.filterwarnings("ignore")
```

#### Configuration Management
Uses `@dataclass` decorators for configuration:
```python
from dataclasses import dataclass, field

@dataclass
class GlobalConfig:
    features: list = field(default_factory=lambda: feature_dict[feature])
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42
```

#### Pandas Operations
- Use `apply()` with lambda functions for row-wise operations
- Be aware of `SettingWithCopyWarning` (occurs in data preprocessing)
- Use `.loc[]` for proper DataFrame indexing

---

## Key Components

### 1. Data Preprocessing Pipeline

#### Location
`src/data scripts/data_preprocess.ipynb`

#### Core Functions

**`calculate_overlap(premise, hypothesis)`**
- Tokenizes both sentences
- Calculates word overlap using Counter intersection
- Returns overlap count

**`detect_word_overlap_bias(df)`**
- Adds `overlap` column to DataFrame
- Adds `is_word_overlap` boolean column
- Returns modified DataFrame

**`extract_negation(tokens)`**
- Checks tokens against negation word list
- Returns list of found negation words

**`detect_subsequence(premise, hypothesis)`**
- Removes punctuation and lowercases text
- Uses regex with word boundaries
- Returns tuple: (hypothesis, is_match)

**Negation Words List**
```python
negation_words = ["not", "no", "n't", "none", "neither", "never",
                  "nobody", "nothing", "nowhere", "hardly",
                  "scarcely", "barely", "rarely", "seldom"]
```

### 2. Model Training Infrastructure

#### Location
`src/evaluation/Auto_X01_BERT_coliee_models_w_features.ipynb`

#### Configuration System

**Feature Dictionary**
```python
feature_dict = {
    "SENTENCE_LENGTH": ['hyp_length'],
    "WORD_OVERLAP": ['overlap', 'is_word_overlap'],
    "HAS_CONTRADICTION_WORDS": ['has_negation'],
    "SUBSEQUENCE_HEURISTICS": ['is_subsequence_heuristic'],
    "ALL": 'all',
    "NONE": "None"
}
```

**Model Dictionary**
Maps friendly names to HuggingFace model identifiers.

#### Custom Dataset Classes

**`CustomDataset`**
- Inherits from `torch.utils.data.Dataset`
- Extracts selected features based on configuration
- Appends features to hypothesis with `[SEP]` token
- Tokenizes premise + hypothesis pairs
- Max length: 512 tokens for full-context

**`CustomHypOnlyDataset`**
- Variant that only uses hypothesis text
- Useful for testing hypothesis-only bias
- Max length: 180 tokens

#### Training Process

1. **Data Split**: 90% train, 10% validation (from train set)
2. **Model Loading**: AutoModelForSequenceClassification
3. **Training Arguments**:
   - Batch size: 8
   - Gradient accumulation: 4 steps
   - FP16 enabled
   - Evaluation every epoch
   - Early stopping (optional)
4. **Evaluation**: Both normal and adversarial test sets
5. **Logging**: Results saved to text files and W&B

### 3. Evaluation Metrics

#### Primary Metric
- **Accuracy**: Used for model selection and evaluation

#### Adversarial Evaluation
Custom functions calculate accuracy for:
- **By Artefact Type**: Word Overlap, Contradiction Word, Annotation Artefacts
- **By Direction**: "For" (exploiting artefact) vs "Against" (contradicting artefact)

#### Functions
- `calculate_separate_acc(dataset, adv_type)`: Filter by adversarial type
- `calculate_artefact_type_acc(dataset, artefact_type)`: Filter by artefact category

---

## Data Pipeline

### Input Data Structure

The data folder (not in git) should be structured as:
```
data/
└── coliee_data/
    ├── processed/
    │   └── task 4/
    │       ├── train/
    │       │   ├── coliee_train_2018.csv
    │       │   ├── coliee_train_2019.csv
    │       │   ├── coliee_train_2020.csv
    │       │   ├── coliee_train_2021.csv
    │       │   └── coliee_train_2022.csv
    │       └── test/
    │           ├── coliee_test_2018.csv
    │           ├── coliee_test_2019.csv
    │           ├── coliee_test_2020.csv
    │           ├── coliee_test_2021.csv
    │           └── coliee_test_2022.csv
    └── task4/
        └── adv-task4/
            └── test/
                └── adversarial_test_set.csv
```

### Processing Steps

1. **Raw Data → Parsed Data** (`data_parse.ipynb`)
   - Extract premise-hypothesis pairs from COLIEE XML/JSON files
   - Create basic CSV structure

2. **Parsed Data → Preprocessed Data** (`data_preprocess.ipynb`)
   - Add label mapping (Y→1, N→0)
   - Tokenize sentences
   - Calculate features (overlap, negations, subsequences)
   - Save with all feature columns

3. **Preprocessed Data → Augmented Data** (`src/mitigation/`)
   - Generate synthetic instances to balance artefacts
   - Validate augmentations for quality
   - Combine with original data
   - Save as `coliee_aug_train_{year}.csv`

4. **Training Data → Model** (`src/evaluation/`)
   - Load processed CSV
   - Apply feature selection
   - Train transformer models
   - Save checkpoints

5. **Model → Predictions** (`src/evaluation/`)
   - Evaluate on test sets
   - Generate instance-level predictions
   - Calculate aggregate metrics
   - Save results to CSV and text files

---

## Important Notes

### Security and Credentials

⚠️ **API Keys**: The codebase previously contained hardcoded API keys (W&B key visible in notebooks). When working with this codebase:
- **NEVER commit API keys to git**
- Use environment variables: `os.environ.get('WANDB_API_KEY')`
- Use `wandb login` CLI command instead of hardcoding

### File Paths

#### Absolute vs Relative Paths
The notebooks use various path construction methods:
```python
# Dynamic path construction (preferred)
folder_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data/coliee_data")

# Hardcoded paths (avoid when editing)
path = "/Users/venkateshmurugadas/software_codes/nli_artefacts_detection/data/..."
```

**When modifying**: Update hardcoded paths to use relative paths or environment variables.

#### Google Colab Specifics
Some notebooks assume Colab environment:
- Google Drive mounting: `/content/drive/MyDrive`
- Google Cloud Storage (gcsfuse): `/content/x01-coliee_dir`
- These need adaptation for local environments

### Data Not Included

The following are required but not in the repository:
1. **COLIEE Dataset**: Must be obtained from COLIEE organizers
2. **Trained Models**: Large model checkpoints are excluded
3. **Plots and Results**: Generated during execution

### Pandas Warnings

The code produces `SettingWithCopyWarning` warnings. These are generally safe to ignore in the notebook context but should be addressed in production code using `.loc[]` indexing.

### Reproducibility

**Random Seed**: Set to 42 throughout the codebase for reproducibility
```python
SEED = 42
set_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
```

### GPU Requirements

- Training scripts assume CUDA availability
- Fallback to CPU exists but training will be very slow
- FP16 training enabled by default (requires GPU)

---

## Common Tasks

### Task 1: Add New Features

**Location**: `src/data scripts/data_preprocess.ipynb`

1. Define feature extraction function
2. Add to preprocessing pipeline
3. Update `feature_dict` in evaluation scripts
4. Regenerate preprocessed datasets

### Task 2: Train Model on New Year

**Location**: `src/evaluation/Auto_X01_BERT_coliee_models_w_features.ipynb`

1. Ensure data files exist for the new year
2. Add year to `years` list
3. Run notebook (it loops through all years)

### Task 3: Add New Model Architecture

**Location**: `src/evaluation/Auto_X01_BERT_coliee_models_w_features.ipynb`

1. Add model identifier to `models` dictionary
2. Create corresponding config class (e.g., `NewModelConfig`)
3. Update `get_model_config()` function
4. Set `chosen_model` variable

### Task 4: Analyze Results

**Location**: `src/evaluation/results/`

1. Ensure prediction CSVs are generated
2. Run `model_results_analysis.ipynb` for statistics
3. Run `Results_Visualization.ipynb` for plots

### Task 5: Create Adversarial Examples

**Location**: `src/detection/coliee_artefacts_detection.ipynb`

1. Analyze dataset for artefact patterns
2. Manually create adversarial instances
3. Save to `adversarial_test_set.csv`
4. Preprocess with standard pipeline

### Task 6: Run Data Augmentation

**Location**: `src/mitigation/`

1. Run `create_data_augmentation_instances.ipynb`
2. Run `validate_augmented_instances.ipynb`
3. Run `combine_augmented_coliee_datasets.ipynb`
4. Use augmented files in training

---

## Git Workflow

### Current Branch
You are working on: `claude/add-claude-documentation-n6fhn`

### Committing Changes

When ready to commit:
```bash
git add CLAUDE.md
git commit -m "Add comprehensive CLAUDE.md documentation for AI assistants"
git push -u origin claude/add-claude-documentation-n6fhn
```

### Creating Pull Requests

After pushing, create PR with:
```bash
gh pr create --title "Add CLAUDE.md documentation" --body "..."
```

---

## Questions and Support

### For Issues
Report at: https://github.com/anthropics/claude-code/issues

### For Thesis Questions
Contact: Venkatesh Murugadas (see LICENSE file)

### For COLIEE Dataset
Visit: Competition on Legal Information Extraction and Entailment website

---

## Changelog

### 2026-01-20
- Initial CLAUDE.md creation
- Comprehensive repository analysis
- Documentation of structure, workflows, and conventions

---

**End of CLAUDE.md**
