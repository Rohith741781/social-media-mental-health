# Social Media & Mental Health: ML Impact Analysis

An end-to-end Machine Learning pipeline tracking the clinical impact of screen time variations, sleep, and social comparison triggers across demographics, featuring a Neural Network build and Professional Streamlit Dashboard.

## Setup Instructions

1. **Install Python Requirements:**
   Download dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute Full ML Pipeline:**
   Generate all data splits, train the Logistic Regression, Random Forest, XGBoost, and Neural Network models, map K-Means clusters + Association rules, and natively export the weights to `/output`.
   ```bash
   python master_pipeline.py
   ```
   *Note: This command will create `.pkl` models and `.png` exploratory plots locally.*

3. **Launch the Dashboard:**
   Render the fully styled frontend via Streamlit to view data insights and Plotly graphs in your browser:
   ```bash
   streamlit run app.py
   ```

## Directory Structure
- `master_pipeline.py` - Core ML logic, model training, metric execution.
- `app.py` - Plotly-backed Streamlit interactive application.
- `dashboard.html` - Static, standalone Plotly.js visual dashboard.
- `output/` - Contains saved joblib models and PNG metric charts.

## Insights
This module discovers that while overall screen volume heavily triggers male depression onset, specific *Social Comparison* triggers and *Late Night Usage* heavily isolate and compound female anxiety clusters. Explore `dashboard.html` or the Streamlit app for Executive Recommendations.
