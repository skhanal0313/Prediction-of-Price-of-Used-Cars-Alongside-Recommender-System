
# Used Car Price Prediction & Recommender System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-orange)

## 🚗 Project Overview

This project presents a machine learning pipeline to **predict the price of used cars** and provide a **personalised recommender system**. Leveraging a large dataset scraped from Craigslist (426,000+ listings), the system helps users make informed decisions when buying or selling vehicles by delivering accurate price predictions and tailored suggestions.

## 📊 Dataset

- **Source**: [Kaggle - Craigslist Car/Truck Data](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
- **Size**: ~426,880 entries, 26 features
- **Attributes**: price, manufacturer, model, year, odometer, fuel type, cylinders, transmission, condition, etc.

## 💡 Key Features

- 🔍 **Exploratory Data Analysis**: Distribution analysis, outlier handling, feature correlations
- 🧹 **Data Preprocessing**: Filtering, imputation, encoding, scaling, log transformation
- 🚀 **ML Models Used**:
  - Linear Regression
  - Random Forest Regressor (Best Performing)
  - XGBoost Regressor
  - K-Nearest Neighbors (KNN)
- ⚙️ **Model Evaluation Metrics**:
  - R² Score
  - Root Mean Squared Error (RMSE)
  - Cross-validation with hyperparameter tuning
- 🤖 **Recommender System**:
  - Interactive input for car attributes
  - Real-time price suggestions based on trained ML models

## 🛠️ Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/your-username/used-car-price-predictor.git
cd used-car-price-predictor
```

### 2. Set Up Environment
Ensure you have Python 3.8+ and install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run Notebook
Open and run the notebook to:
- Clean data
- Train models
- Evaluate results
- Test the recommender system

```bash
jupyter notebook PRML_ProjCode-GPNo47.ipynb
```

## 👥 Contributors

- **Rohit Baral** – u3268702@uni.canberra.edu.au  
- **Sujan Khanal** – u3258630@uni.canberra.edu.au

## 📄 License

This project is licensed under the MIT License.

## 📚 References

- [Craigslist Vehicle Dataset on Kaggle](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
- Refer to the full [Project Report](./PRMLProjReport-GpNo2.pdf) for detailed methodology and analysis.
