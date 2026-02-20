# Financial Fraud Risk System

## Overview

This repository contains a structured, end-to-end fraud risk system prototype designed for financial services environments including banking, fintech, payments, and lending.

The project simulates an enterprise-grade fraud detection workflow, including:

- Synthetic transaction generation (300,000 records)
- Enterprise-style feature engineering (50 risk features)
- Stratified model training pipeline
- Class imbalance handling
- Modular system structure for scalability

This repository focuses on system design, reproducibility, and professional engineering practices rather than notebook-only experimentation.

---

## Objectives

The goal of this project is to demonstrate:

- Practical fraud modeling workflow
- Risk-aware data engineering
- Class imbalance handling in fraud detection
- Clean project structuring for production-readiness
- Separation between raw, processed, and modeling layers

This system is designed to reflect how fraud risk models are developed and evaluated in financial institutions.

---

## System Architecture

Transaction Data  
→ Feature Engineering  
→ Train/Test Split (Stratified)  
→ Model Training (Logistic Regression baseline)  
→ Evaluation (Precision, Recall, ROC-AUC)  

The structure separates:

- Data generation
- Feature transformation
- Model training
- Evaluation logic

---

## Project Structure


---

## Dataset

The dataset includes:

- 300,000 synthetic transactions
- 50 enterprise-style fraud risk features
- ~5–6% fraud rate
- Transaction, velocity, device, account, geo, and behavioral signals

Raw datasets are excluded from the repository.

To regenerate synthetic data:


---

## Modeling Approach

Baseline model:

- Logistic Regression
- Class imbalance handled using `class_weight="balanced"`
- Stratified train/test split

Evaluation metrics:

- Precision
- Recall
- F1-score
- ROC-AUC

The focus is on fraud-relevant metrics rather than accuracy.

---

## Design Principles

This project follows several production-oriented principles:

- Reproducibility via fixed random seeds
- Explicit feature engineering separation
- No data leakage
- No identifiers used as features
- Clear raw vs processed data layers
- Modular and scalable code structure

---

## Future Enhancements

Planned extensions include:

- Random Forest and XGBoost models
- Threshold optimization based on cost function
- Fraud spike simulation
- Model drift monitoring
- Hybrid rules + ML architecture
- Business cost optimization layer

---

## Requirements

Install dependencies:


---

## Disclaimer

All data in this repository is synthetic and generated for educational and demonstration purposes. No real customer or financial data is used.

---

## Author

Developed as part of a structured fraud risk systems portfolio project focused on financial services risk modeling and architecture design.
