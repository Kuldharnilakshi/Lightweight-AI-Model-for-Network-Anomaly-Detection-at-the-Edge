# Evaluation report

## Dataset
- Synthetic dataset with four classes: benign, DDoS, spoofing, portscan.
- Rows: N per class. See `data/dataset.csv`.

## Models trained
- RandomForest baseline (`models/rf_baseline.pkl`)
- Keras MLP (`models/keras_full.h5`) -> TFLite (`models/edge_model.tflite`)

## Metrics (example)
- RF baseline CV accuracy: X.XX
- MLP test accuracy: X.XX
- Per-class F1 scores: see `notebooks/eda_and_training.ipynb` outputs.

## Edge benchmarks (example)
- TFLite inference latency mean: 2.7 ms (on laptop)
- TFLite inference latency mean: 18.4 ms (on Raspberry Pi 4)  <-- run your own

## How to reproduce
1. Generate dataset: `python src/generate_dataset.py --n 800 --out data/dataset.csv`
2. Train & evaluate: `python src/train_cv.py --data data/dataset.csv --out models --epochs 20`
3. Export TFLite: `python src/export_tflite.py --keras models/keras_full.h5 --scaler models/scaler_keras.pkl --out models/edge_model.tflite`
4. Run simulator: `python src/inference_simulator.py --data data/dataset.csv --model models/edge_model.tflite --scaler models/scaler_keras.pkl`
