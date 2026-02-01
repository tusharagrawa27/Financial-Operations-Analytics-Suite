# =====================================================================================================
# PROFITABLITY ANALYSIS AND COHORT ANALYSIS.......
# =====================================================================================================

# IMPORTING ALL IMPORTANT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


ft = pd.read_csv("financial_transactions.csv")
print("=====================================================\nPRINTING ALL THE COLOUMS OF FINANCIAL_TRANSACTION\n=====================================================")
# ALL COLOUMS
print(ft.columns)

fc = pd.read_csv('financial_customers.csv')
# ALL COLOUMS
print("\n\n\n============================================\nALL THE COLOUMS OF FINANCIAL_CUSTOMERS \n============================================")
print(fc.columns)

ft['transaction_date'] = pd.to_datetime(ft['transaction_date'])
# IN ft CREATING A NEW COLOUM THAT GIVE ONLY MONTH OF TRANSCATION 
ft['transaction_month'] = ft['transaction_date'].dt.to_period('M')
df = ft.copy()
# =====================================================================================================
# RETENTION RATE ANALYSIS                                                                             |
# =====================================================================================================
df['cohort_year'] = df['cohort_month'].astype(str).str[:4].astype(int)
df['cohort_month_num'] = df['cohort_month'].astype(str).str[5:7].astype(int)

df['trans_year'] = df['transaction_month'].astype(str).str[:4].astype(int)
df['trans_month_num'] = df['transaction_month'].astype(str).str[5:7].astype(int)

# Cohort Period calculate karein (Months ka gap)
df['cohort_period'] = (df['trans_year'] - df['cohort_year']) * 12 + \
                      (df['trans_month_num'] - df['cohort_month_num'])
# --- NAYA CODE END ---

# Step 4: Har cohort aur period ke liye unique customers count karein
cohort_data = df.groupby(['cohort_month', 'cohort_period'])['customer_id'].nunique().reset_index()

# Step 5: Data ko pivot table mein badlein taaki matrix ban sake
cohort_counts = cohort_data.pivot(index='cohort_month', columns='cohort_period', values='customer_id')

# Step 6: Retention Rate calculate karein
cohort_sizes = cohort_counts.iloc[:, 0]
retention_matrix = cohort_counts.divide(cohort_sizes, axis=0)

# Step 7: Heatmap plot karein
plt.figure(figsize=(14, 10))
sns.heatmap(retention_matrix, annot=True, fmt='.0%', cmap='YlGnBu', vmin=0.0, vmax=1.0)
plt.title('Customer Retention Rates by Monthly Cohorts')
plt.xlabel('Months Since Acquisition (Cohort Period)')
plt.ylabel('Cohort Month')
plt.show()

# Save karna ho toh:
retention_matrix.to_csv('retention_rates.csv')
print("Retention Rate calculation complete!")