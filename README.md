# EKG-Classifier  
*A Domain-Adversarial Deep Learning Pipeline for Multi-Label ECG Classification (PTB-XL + MIT-BIH)*

This repository contains a full training and inference pipeline for classifying electrocardiogram (ECG) signals using a **Domain-Adversarial Neural Network (DANN)**.  
The system is designed to learn from the PTB-XL dataset while adapting to the MIT-BIH Arrhythmia Database, improving cross-dataset generalization.

The project includes:

- Data preprocessing (lead normalization, resampling, filtering, z-scoring)  
- 12-lead mapping for PTB-XL and MIT-BIH  
- A ResNet-based feature extractor  
- A multi-label classification head  
- A domain classifier with Gradient Reversal (GRL)  
- Training and evaluation scripts  
- A live demo script capable of running:
  - Test samples from processed datasets  
  - **External `.npz` or `.npy` ECG files**

---

## Features

- **Multi-label ECG classification**  
  Supports fine-grained SCP codes from PTB-XL.

- **Domain adaptation with DANN**  
  Reduces domain shift between PTB-XL and MIT-BIH.

- **Fast preprocessing pipeline**  
  Includes bandpass filtering, resampling, and lead normalization.

- **High-throughput DataLoaders**  
  Persistent workers, pinned memory, and prefetching.

- **Inference demo**  
  Run the trained model on:
  - Any test index in the dataset  
  - Any standalone `.npz` or `.npy` file

---

## Installation

This project uses **Python 3.11** and **Poetry** for dependency management.

### 1. Clone the repo

```bash
git clone https://github.com/dfigueredo0/EKG-Classifier
cd EKG-Classifier
```

### 2. Install dependencies
```bash
poetry install
```

### 3. Activate the environment
```bash
poetry activate
```

---

## Dataset Setup
The project expects PTB-XL and optionally MIT-BIH Arrhythmia Database.

PTB-XL

Download from PhysioNet:
https://physionet.org/content/ptb-xl/1.0.3/

Place the extracted dataset under:
`data/raw/ptbxl/`

MIT-BIH Arrhythmia

Download:
https://physionet.org/content/mitdb/1.0.0/

Place under:
`data/raw/mitbih/`

---

## Preprocessing

Before training, convert raw WFDB records into normalized .npz files.

PTB-XL
```bash
poetry run python -m ekgclf.data.make_dataset `
    --raw data/raw/ptbxl `
    --out data/processed `
    --dataset ptbxl `
    --config configs/data.yaml
```

MIT-BIH

MIT-BIH requires a CSV defining window slices (start/end times).
Once prepared:
```bash
poetry run python -m ekgclf.data.make_dataset `
    --raw data/raw/mitbih `
    --out data/processed `
    --dataset mitbih `
    --config configs/data.yaml
```

---

## Training the Model

Train the Domain-Adversarial Neural Network:

```bash
poetry run python -m ekgclf.train_dann
```

Training configuration is stored in:

`configs/train_dann.yaml`
`configs/model.yaml`
`configs/data.yaml`

The best checkpoint is saved to:

`checkpoints/dann/best_dann.pt`

You can adjust:
- Learning rate
- Batch size
- Number of epochs
- Weight decay
- Whether to use class weights
- Lambda schedule for gradient reversal

All inside `configs/train_dann.yaml`.

---

## Demo using dataset indices

Each processed ECG file is listed in ptbxl_index.json or mitbih_index.json.

Run inference on any entry:
```bash
poetry run python -m ekgclf.run_dann_demo --idx 1234
```

Output includes:

- Domain prediction (PTB-XL vs MIT-BIH)
- Top-5 predicted diagnostic labels
- Ground-truth labels
- Confidence scores

---

## Demo using an external .npz or .npy file

Your file must contain ECG data shaped: 

    [T, C]  # time × channels

If `.npz`, it must include: 

    signals = <array [T, C]>

Example:
```bash
poetry run python -m ekgclf.run_dann_demo \
    --file examples/sample_ekg.npz
```

or:
```bash
poetry run python -m ekgclf.run_dann_demo \
    --file examples/custom_ecg.npy
```

The model will:
- Infer domain (PTB-XL-like or MIT-BIH-like)
- Predict ECG diagnostic labels
- Print top confidence scores

---

## License
This project is open-source under the MIT license

---

## Support/Contributions
Open an issue or pull reuqest if you'd like to contribute, request enchancements, or extend functionality. 
