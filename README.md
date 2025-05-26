# STORM‑AI Hybrid Residual – Minimal Guide

## 1 · Create the environment

```bash
conda env create -f environment.yml -n storm-ai
conda activate storm-ai
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

