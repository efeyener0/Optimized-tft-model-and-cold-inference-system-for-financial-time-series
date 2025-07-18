# Production-Ready TFT Pipeline for Financial Time Series Forecasting

⚠️ Disclaimer: Not Financial Advice
This project and all its associated code, models, and data are provided for educational, research, and informational purposes only. They are not, and should not be considered, financial, investment, or trading advice.
The models and predictions generated by this software are the result of statistical processes and are not guaranteed to be accurate. Past performance is not indicative of future results. Financial markets are inherently unpredictable, and trading involves a high level of risk.
You are solely responsible for your own investment decisions and any resulting profits or losses. The author of this project assume no liability for any actions you take based on the information or code provided herein. Always conduct your own thorough research and consult with a qualified financial advisor before making any investment decisions
This repository contains an end-to-end, production-ready, and high-performance Temporal Fusion Transformer (TFT) pipeline designed for financial time series forecasting. The primary objective is not just to train a model, but to establish a robust MLOps framework that makes the entire process **reproducible, reliable, and deployable**.

The core philosophy of this project is to merge the engineering discipline required by modern AI systems with advanced signal processing techniques tailored for financial data.

## 🌟 Key Features & Highlights

*   **End-to-End Architecture:** Covers all critical stages, from data processing and feature engineering to model training, packaging, and inference.
*   **Advanced Feature Engineering:** Utilizes not only standard indicators (EMA, RSI) but also advanced techniques like the **Stationary Wavelet Transform (SWT)** to uncover hidden patterns and frequency components.
*   **Custom Hybrid Loss Function:** Implements a `HybridFinancialLoss` function that incorporates **directional** and **wavelet** components, enabling the model to learn the market's momentum and frequency structure.
*   **Production-Oriented Training (`trainer.py`):**
    *   Features GPU optimizations like `torch.compile` and automatic TF32 activation.
    *   Generates reliable, deployment-ready artifacts (`best_model_weights.pth` and `inference_params.yaml`).
*   **Dual Inference Architectures:** The project deliberately includes two distinct inference logics, showcasing two different engineering philosophies for model deployment—one for rapid testing and one for mission-critical production.

## 🏛️ Architectural Philosophy: Dual Inference Logics

A key feature of this repository is its demonstration of two separate, purpose-built approaches to inference. This reflects a mature understanding that the needs of a developer during experimentation are different from the needs of a system in production.

### 1. The Rapid Prototyping Core (`core/tft_pipeline_core.py`)

This approach is designed for **agility and rapid testing**.

*   **How it Works:** It provides a "cold-start" inference pipeline that works directly from the two essential artifacts: the `inference_params.yaml` configuration and the `.pth` model weights file. It rebuilds the model architecture in memory based on these files.
*   **Use Case:** This is the ideal path for developers. It allows for quick validation of a newly trained model, debugging, or running "what-if" scenarios without the overhead of creating a full deployment package. If you just want to see if your new model works, this is the fastest way.
*   **Engineering Philosophy:** **Speed and Convenience.** Prioritizes developer velocity and iterative development cycles.

### 2. The Production-Grade Packager (`packager.py`)

This approach is built on a **"zero-trust" philosophy for maximum reliability and reproducibility.**

*   **How it Works:** The `packager.py` script creates a completely self-contained, hermetic `deployment_package`. This package includes the model, all necessary source code dependencies (copied locally), and a frozen `requirements.txt` file that locks the exact versions of all libraries used during training.
*   **Use Case:** This is the **recommended path for any real deployment**, whether it's staging, production, or sharing with a colleague. It guarantees that the model will run identically, regardless of the environment, completely eliminating the "it works on my machine" problem. The auto-generated `predictor.py` inside the package is the stable, trusted entry point.
*   **Engineering Philosophy:** **Robustness and Reproducibility.** Prioritizes system stability, eliminates environmental drift, and ensures that what was tested is exactly what gets deployed.

## 🚀 Quick Start Guide (Production Path)

This guide demonstrates the recommended production path.

### 1. Installation

First, install the required libraries. It is highly recommended to use a virtual environment.
```bash
pip install torch pandas numpy pyyaml scikit-learn pytorch-lightning pytorch-forecasting PyWavelets
```

### 2. Configuration

All aspects of the project are controlled via the `config.yaml` file. Edit this file to match your dataset, model parameters, and training goals.

### 3. Training

Start the training process by running the `trainer.py` script.
```bash
python trainer.py --config /path/to/your/config.yaml
```
This will generate `best_model_weights.pth` and `inference_params.yaml` in the `logs/<project_name>/<run_name>/` directory.

### 4. Packaging

Package your trained model into a production-ready artifact.
```bash
python packager.py --config /path/to/your/config.yaml --run_dir logs/<project_name>/<run_name>/
```
This command will create a new, self-contained directory named `deployment_package`.

### 5. Inference

Making predictions with the packaged model is simple and reliable:
```bash
# 1. Navigate to the package directory
cd deployment_package

# 2. (Recommended) Create a new virtual environment and install locked dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Run the stable predictor script
python predictor.py
```
The `predictor.py` script provides a live example of how to use the deployed model and can be easily integrated into your own applications.

## 🧠 Motivation

This project is built on the philosophy that modern AI applications, especially in high-stakes domains, cannot consist of theoretical models alone. A successful system must embrace core MLOps principles: robust engineering practices, testability, reproducibility, and production-readiness. This repository aims to serve as a reference implementation of these principles in practice.
