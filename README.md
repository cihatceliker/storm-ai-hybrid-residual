# STORM‑AI Hybrid Residual – Minimal Guide

## 1 · Create the environment

```bash
conda create -n storm-ai python=3.10 anaconda
conda activate storm-ai
conda install -c conda-forge orekit=12.1.2
pip install -r requirements.txt
pip3 install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy==1.24.3
```

## 2 · Put the raw data in place

```
project_root/
└── data/
    ├── omni2/            # hourly OMNI‑2 CSV files
    ├── sat_density/      # orbit‑mean density targets (public train)
    └── initial_states.csv
```

## 3 · Build the training set

```bash
python prepare_train_data.py
# ⇢ creates cache/ … processed data ready for model training
```

## 4 · Train the models

```bash
python train.py            # checkpoints + helper files saved to assets/
```

## 5 · Prepare the inference package

```bash
cp -r assets/ inference/
# inference/ now matches the exact folder submitted to the competition
```

