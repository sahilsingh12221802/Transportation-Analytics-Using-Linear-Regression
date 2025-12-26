# Transportation Analytics using Linear Regression (PySpark)

This project applies **Linear Regression** using **Apache Spark (PySpark)** to analyze transportation data and predict commuter satisfaction. The goal is to understand how factors such as travel time, distance, income, fuel cost, and transport mode influence satisfaction, and to present insights through a Tableau dashboard.

---

## Project Overview
- Course: Cluster Programming  
- Technique Used: Linear Regression (Supervised Learning)  
- Tools & Technologies:
  - Apache Spark (PySpark)
  - Python (Pandas)
  - Tableau
  - Excel / CSV data sources

---

## Repository Structure
├── Transportation data.xlsx # Original dataset
├── Transportation_Data.csv # Cleaned CSV used in PySpark
├── LinearRegression_Results.csv # Actual vs Predicted output
├── DASHBOARD.twbx # Tableau Dashboard file
├── README.md # Project documentation


---

## Dataset Description
The dataset contains transportation-related attributes such as:
- Demographics: Age, Gender, Occupation, Monthly Income
- Travel Metrics: Travel Distance, Travel Time, Fuel Cost
- Preferences: Transport Mode, Vehicle Ownership, Purpose of Travel
- Target Variable: **Satisfaction_Score**

---

## Methodology
1. Loaded and cleaned the dataset using PySpark.
2. Performed exploratory data analysis (EDA).
3. Encoded categorical variables using `StringIndexer`.
4. Built a Linear Regression model using Spark MLlib.
5. Evaluated performance using RMSE, MAE, and R².
6. Tuned hyperparameters to improve model accuracy.
7. Exported results for visualization in Tableau.

---

## Model Performance
- Metrics Used:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² Score (Accuracy Indicator)
- The tuned model achieved a strong R² score, indicating good predictive performance.

---

## Dashboard
An interactive Tableau dashboard (`DASHBOARD.twbx`) visualizes:
- Satisfaction trends by city type and transport mode
- Travel time vs satisfaction
- Income vs satisfaction
- Actual vs predicted satisfaction scores

---

## Key Insights
- Longer travel time and distance reduce commuter satisfaction.
- Higher income and private vehicle ownership increase satisfaction.
- Urban areas show higher average satisfaction than rural regions.

---

## Conclusion
This project demonstrates how **PySpark** can be effectively used for large-scale transportation analytics and how regression models can support data-driven decision-making. Tableau enhances the interpretability of results through interactive visualizations.

---

