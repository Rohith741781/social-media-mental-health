# 🧠 Social Media & Mental Health Analytics
### **Advanced Machine Learning Pipeline & 3D Interactive Demographic Dashboard**

![Python](https://img.shields.io/badge/Python-3.11-Blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_UI-FF4B4B?style=for-the-badge&logo=streamlit)
![Three.js](https://img.shields.io/badge/Three.js-3D_Physics-000000?style=for-the-badge&logo=three.js)

---

## 📖 Project Overview
This repository functions as a complete end-to-end Machine Learning ecosystem engineered to process, analyze, and map the psychological impacts of Social Media consumption across **7,163 active global users**. 

Connecting raw data points such as `Screen_Time`, `Late_Night_Usage`, and `Social_Comparison_Triggers` against clinical benchmarks like **GAD-7 (Anxiety)** and **PHQ-9 (Depression)**, this proprietary dataset tracks exactly *how* digital exposure correlates strictly to mental degradation separated by Male vs Female demographics.

---

## 🏗️ Technical Architecture

This project is divided into three major infrastructural segments mapping from deep-backend algorithms to responsive front-end 3D graphics:

### 1. **Data Engineering & Feature Extraction**
- **Clinical Thresholding:** Programmatically maps `High_Risk` states (Target Variable = `1`) whenever a user flags `GAD_7_Score >= 10` or `PHQ_9_Score >= 10`.
- **Ratio Engineering:** Computes proprietary inputs such as the `Screen_Sleep_Ratio` and specific `Risk_Interaction` matrices.
- **Stratification Engine:** Employs explicit Gender-Bias controls utilizing Scikit-learn's `train_test_split` algorithms to balance demographics precisely before training.

### 2. **Machine Learning & Deep Learning Topologies**
- **Scikit-Learn Baselines:** Automated parameter-tuning trains and evaluates `RandomForest`, `XGBoost`, `DecisionTree`, and `LogisticRegression` against highly customized Recall-prioritized metrics.
- **Unsupervised Discovery:** Employs K-Means Clustering integrated with PCA dimensionality reduction to group underlying 'Behavioral Personas' visually into 3D spaces.
- **Deep Neural Networks:** PyTorch-integrated logic implementing Batch Normalization, Dropout optimizations, and Early Stopping criteria over customized Epoch arrays evaluating continuous clinical scores.

### 3. **Interactive 3D Deployment Platforms**
- **Streamlit Local Server (`app.py`):** Hooks the `.pkl` / `.pth` serialized pipelines directly into a sleek, modular web application displaying multi-variable tracking for Clinical Triage predictions.
- **Standalone 3D Environment (`index.html`):** The "Wow-Factor" implementation natively bridging HTML, CSS, and `Three.js` (along with `GSAP` physics) dropping users directly into immersive 3D spatial plots comparing gender-driven metrics via `Chart.js` and `Plotly.js` architectures natively without external frameworks.

---

## 📊 Key Clinical Insights & Results

### 🔬 **Comparative Demographic Discovery**
- **Female Cohorts:** Feature importance dictates that **Social Comparison** and visual-centric algorithms (Instagram/TikTok feeds) map linearly to high clinical anxiety (GAD-7 > 10). Females recorded a **28.9% High Risk Factor**, requiring algorithmic visual-blocking interventions.
- **Male Cohorts:** Feature importance indicates that **Aggregate Screen Volume** compounded by **Late Night Deprivation** sequences spikes isolation-tracked depression markers (PHQ-9). Males sit at a systemic **27.5% High Risk Factor** due to sustained volumetric consumption structures.
- **Pattern Matching (Apriori Rules):** Deep Association Rule Mining discovered robust dependencies between nocturnal gaming states and absolute clinical distress factors inside Male-dominant clusters.

---

## 🚀 Installation & Execution Instructions

### Prerequisites
Make sure you have **Python 3.11+** installed alongside native Git operations.

### 1. Repository Setup
```bash
# Clone the infrastructure
git clone https://github.com/Rohith741781/social-media-mental-health.git
cd social-media-mental-health

# Create and sync virtual environments
pip install -r requirements.txt
```

### 2. Run the Machine Learning Pipeline
To retrain models from scratch, execute the primary logic pipeline. This generates new analytical plots natively into the `/output/` folder and saves newly compiled states.
```bash
python mental_health_pipeline.py
```

### 3. Deploy the Streamlit Predictive UI
To interface with the data intelligently through a Python-based UX environment:
```bash
streamlit run app.py
```

### 4. Experience the Immersive 3D Space
To access the ultra-premium Three.js and Plotly interactive Web App, natively open the bundled index file in your browser, or quickly spin up a background server:
```bash
python -m http.server 8080
# Navigate to http://localhost:8080/index.html
```

---

## 📁 Repository Structure

```tree
social-media-mental-health/
│
├── README.md                          # Primary Project Documentation
├── requirements.txt                   # Dependency Definitions
├── index.html                         # Premium 3D Standalone Interactive SPA
├── app.py                             # Modular Streamlit Logic Host
├── master_pipeline.py                 # Core Analytics Engine Logic
├── mental_health_pipeline.py          # Detailed Machine Learning Implementations
├── social_media_mental_health.xlsx    # Local Project Database
│
└── output/                            # Extracted Renderings and Models
    ├── models/                        # Serialized .pkl and PyTorch .pth topologies
    ├── plots/                         # Advanced Matplotlib visual PNGs
    └── metrics/                       # JSON performance analytics
```

---

*This application was engineered using state-of-the-art intelligent environments designed to systematically deploy enterprise visualization frameworks seamlessly.*
