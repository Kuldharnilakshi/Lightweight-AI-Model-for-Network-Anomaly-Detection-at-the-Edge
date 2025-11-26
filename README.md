# Lightweight AI Model for Network Anomaly Detection (Prototype)

This is a prototype that demonstrates a lightweight ML pipeline for edge-based network anomaly detection:
- synthetic dataset generation
- training a tiny Keras model and exporting to TFLite
- TFLite inference simulator (measures latency)
- live capture inference (requires `scapy` and root)

## Setup (Linux / macOS)
```bash
# 1. create venv
python3 -m venv venv
source venv/bin/activate

# 2. install deps
pip install -r requirements.txt
