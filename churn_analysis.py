# =====================================================================================================
# CHURN ANALYSIS
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

# IMPORTING CSV FILE OF FINANCIAL_CUSTOMER.CSV
fc = pd.read_csv('financial_customers.csv')

# ALL COLOUMS
print("============================================\nALL THE COLOUMS OF THE CSV IS HERE \n============================================")
print(fc.columns)
print("\n========================================================================================")

# --------------------------------------------
# PERFORMING CHURN ANALYSIS MODEL
# --------------------------------------------

# MAKING ADDISTIONAL NEW SET OF DATA X AND Y
print("==============================\n THE MODLE IS PREPARING.......\n==============================")
x= fc[['mrr', 'contract_length','number_of_users', 'support_tickets', 'usage_score', 'nps_score', 'recency_days', 'transaction_count', 'avg_transaction_value']]
y = fc['is_churned']

gb = GradientBoostingClassifier()
gb.fit(x, y)

fc['GB_Prob'] = gb.predict_proba(x)[:,1]
fc['GB_Pred'] = gb.predict(x)
print("==============================\n THE MODLE IS PREPARED SUCCESSFULLY\n==============================")

# ------------------------------------------------------------------
# MAKING NEW DATA FRAME FOR customer_id AND GB_Pred
# ------------------------------------------------------------------
future_churn = fc[['customer_id','GB_Pred']]
future_churn['Status_of_churn'] = future_churn['GB_Pred'].apply(lambda x : 'will churn' if x == 1 else 'will not churn')
# converting a csv file for people who can go
future_churn_1 = future_churn[future_churn['Status_of_churn'] == 'will churn' ]
print("==============================\n CSV FOR FUTURE CHURN IS SAVING......\n==============================")
# future_churn_1.to_csv('future_churn_V2.csv', index=False)
print("==============================\n CSV FOR FUTURE CHURN IS SAVED\n==============================")

# -------------------------------
# Groupby: Plan-wise churn count
# -------------------------------
plan_churn = (
    fc
    .groupby('plan')
    .agg(No_of_Churned=('is_churned', lambda x: (x == 1).sum()))
    .reset_index()
)
# -------------------------------
# Groupby: INDUSTRY-wise churn count
# -------------------------------
ind_churn = (
    fc
    .groupby('industry')
    .agg(No_of_Churned=('is_churned', lambda x: (x == 1).sum()))
    .reset_index()
)
# -------------------------------
# Groupby: country-wise churn count
# -------------------------------
country_churn = (
    fc
    .groupby('country')
    .agg(No_of_Churned=('is_churned', lambda x: (x == 1).sum()))
    .reset_index()
)
# -------------------------------
# Groupby: Segment-wise churn count
# -------------------------------
seg_churn = (
    fc
    .groupby('segment')
    .agg(No_of_Churned=('is_churned', lambda x: (x == 1).sum()))
    .reset_index()
)
# --------------------------------------------------------
# PERFORMING RANDOMFORCAST FOR FINDING ROOT OF CHURN
# --------------------------------------------------------
# 1. Dataset Load 
df = pd.read_csv('financial_customers.csv')

# 2. X aur y define
Y = df['is_churned']
X = df[['mrr', 'contract_length', 'number_of_users', 'support_tickets', 
        'usage_score', 'nps_score', 'transaction_count', 'avg_transaction_value','recency_days']]

# 3. Data Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. RANDOM FOREST Model Train 
# n_estimators=100 ka matlab hai hum 100 chote-chote trees (Decision Trees) bana rahe hain
rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# 5. Feature Importance
rf_feat_imp = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance %': (rf.feature_importances_ * 100).round(2)
}).sort_values('Importance %', ascending=False)

print("========================================\nRandom Forest churn causes:\n========================================")
print(rf_feat_imp.to_string(index=False))

# 6. Graph
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance %', y='Feature', data=rf_feat_imp, palette='coolwarm')
plt.title('Real Behavioral Drivers for Churn (Random Forest)')
plt.xlabel('Importance (%)')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
# -------------------------------
# subploting graphs
# -------------------------------

fig, ax = plt.subplots(2, 2,figsize=(10,10))

# -------------------------------
# Subplot (0,0) : Bar Chart
# -------------------------------
bars = ax[0,0].bar(
    plan_churn['plan'], 
    plan_churn['No_of_Churned']
)
# Bar labels
for bar in bars:
    height = bar.get_height()
    ax[0,0].text(
        bar.get_x() + bar.get_width()/2,
        height,
        int(height),
        ha='center',
        va='bottom',
        fontsize=10
    )
# Formatting
ax[0,0].set_title("No. of People Churned by Plan")
ax[0,0].set_ylabel("No. of Churned Customers")
ax[0,0].tick_params(axis='x', rotation=0)
# -------------------------------
# Subplot (1,0) : Bar Chart
# -------------------------------
bars1 = ax[1,0].bar(
    ind_churn['industry'], 
    ind_churn['No_of_Churned']
)
# Bar labels
for bar in bars1:
    height = bar.get_height()
    ax[1,0].text(
        bar.get_x() + bar.get_width()/2,
        height,
        int(height),
        ha='center',
        va='bottom',
        fontsize=10
    )
# Formatting
ax[1,0].set_title("No. of People Churned by industry")
ax[1,0].set_ylabel("No. of Churned Customers")
ax[1,0].tick_params(axis='x', rotation=45)
# -------------------------------
# Subplot (0,1) : Bar Chart
# -------------------------------
bars2 = ax[0,1].bar(
    country_churn['country'], 
    country_churn['No_of_Churned']
)
# Bar labels
for bar in bars2:
    height = bar.get_height()
    ax[0,1].text(
        bar.get_x() + bar.get_width()/2,
        height,
        int(height),
        ha='center',
        va='bottom',
        fontsize=10
    )
# Formatting
ax[0,1].set_title("No. of People Churned by Country")
ax[0,1].set_ylabel("No. of Churned Customers")
ax[0,1].tick_params(axis='x', rotation=0)
# -------------------------------
# Subplot (1,1) : Bar Chart
# -------------------------------
bars3 = ax[1,1].bar(
    seg_churn['segment'], 
    seg_churn['No_of_Churned']
)
# Bar labels
for bar in bars3:
    height = bar.get_height()
    ax[1,1].text(
        bar.get_x() + bar.get_width()/2,
        height,
        int(height),
        ha='center',
        va='bottom',
        fontsize=10
    )
# Formatting
ax[1,1].set_title("No. of People Churned by segment")
ax[1,1].set_ylabel("No. of Churned Customers")
ax[1,1].tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.show()


print("========================================\n ALL THE GRAPHS ARE DRAWAN\n========================================")


# =====================================================================================================
# CUSTOMER ANALYSIS                                                                                   |
# =====================================================================================================

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


print("\n========================================\n CUSTOMER ANALYSIS \n========================================")
# --------------------------------------------------------------
# 1. Deviding customers on VIP,SPECIAL,VALUABLE AND NOT VALUABLE
# --------------------------------------------------------------

# APPLYING KMEAN MODLE TO DO SO
K = fc[['mrr','number_of_users','nps_score','lifetime_months','clv','avg_transaction_value']]
scaler = StandardScaler()
K_scaled = scaler.fit_transform(K)
kmeans = KMeans(n_clusters=4, random_state=42) # random_state fix karne se results consistent rahenge
fc['Segment'] = kmeans.fit_predict(K_scaled)
Customer_segment = fc[['customer_id','mrr','number_of_users','nps_score','lifetime_months','clv','avg_transaction_value']]
Customer_segment['customer_label'] = fc['Segment'].apply(
    lambda x: 'Loyal' if x == 0 else ('Vip' if x == 1 else ('At-Risk' if x == 3 else 'The Enterprise'))
)
print("\n========================================\n printing groupby mean of KMEAN \n========================================")
cluster_segment = Customer_segment.groupby('customer_label')[['mrr','number_of_users','lifetime_months','clv','avg_transaction_value']].mean()
print(cluster_segment)
Customer_segment.to_csv('customer_segment.csv')
print(Customer_segment.head(20))



