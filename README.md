# Financial-Operations-Analytics-Suite
# Financial Operations Analytics Suite 📊

## Project Overview
This project focuses on solving critical business problems for a subscription-based financial service. It utilizes **Python** and **Machine Learning** to analyze customer behavior, predict churn, and forecast future revenue.

## Key Features
* **Revenue Forecasting:** Used **Facebook Prophet** to predict monthly revenue for the next 12 months with trend and seasonality analysis.
* **Churn Prediction:** Built a Machine Learning model (Gradient Boosting/Random Forest) to identify at-risk customers.
* **Customer Segmentation:** Applied **K-Means Clustering** to segment customers into VIP, Loyal, and At-Risk groups based on CLV and usage.
* **Profitability Analysis:** Conducted Cohort Analysis to track retention rates over time.

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Prophet, Matplotlib, Seaborn
* **Tools:** Jupyter Notebook / VS Code

## Key Insights
* Identified that customers with lower usage scores and higher support tickets are 3x more likely to churn.
* Forecasted a 15% growth in revenue for Q1 2026 based on historical trends.
* Customer retention drops significantly after the 3rd month, indicating a need for better onboarding.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install pandas sklearn prophet matplotlib seaborn`
3. Run `data_cleaning.py` first, followed by `churn_analysis.py` or `revenue_forcasting.py`.

---
*Created by Tushar Agrawal | Mechanical Engineering Undergrad & Data Analyst Aspirant*
