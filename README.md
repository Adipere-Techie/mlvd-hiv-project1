# 📊 HIV Mortality Prediction & Tableau Dashboard

This project uses real-world HIV data from UNAIDS to predict AIDS-related deaths using machine learning and visualize the results with Tableau.

## 📁 Contents
- `hiv_model_random_forest.py` – Python script for merging data, training a Random Forest model, and exporting results
- `Merged_HIV_Dataset.xlsx` – Cleaned and merged dataset used for training and visualization
- `RF_Predictions_HIV.xlsx` – Model predictions ready for Tableau
- `MLVD_Dashboard.twbx` – Tableau dashboard with filters, charts, and What-If simulations

## 📈 Model Summary
- **Model:** Random Forest Regressor
- **R² Score:** 0.998
- **RMSE:** ~563.46
- **Top Features:** Region, Year, ART Coverage

## 📊 Tableau Dashboard Features
- Line chart: Deaths over time by region
- Scatter plot: Actual vs Predicted deaths
- Feature importance bar chart
- What-If simulation with parameter sliders
- Filters for region and year

## 📦 Tools Used
- Python (pandas, scikit-learn, matplotlib)
- Excel
- Tableau Public

## 🌍 Data Source
All datasets were sourced from: [https://aidsinfo.unaids.org/](https://aidsinfo.unaids.org/)

## 🧾 Author
Adipere Gift Feateide – MSc Information Technology Management (2025)
