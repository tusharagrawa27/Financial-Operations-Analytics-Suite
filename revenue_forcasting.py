import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from prophet import Prophet
import matplotlib.pyplot as plt

# -------------------------------
# 1️ Data Load & Preparation
# -------------------------------

mr = pd.read_csv('monthly_revenue.csv')
df = mr.copy()

# Date column ko datetime me convert kar rahe hain
df['year_month'] = pd.to_datetime(df['year_month'], format='%d-%m-%Y')

# Prophet ke liye mandatory columns
df['ds'] = df['year_month']      # date
df['y'] = df['total_revenue']    # target variable

# -------------------------------
# 2️ Prophet Model Training
# -------------------------------

model = Prophet()
model.fit(df)

# Future ke 12 months ka dataframe
future = model.make_future_dataframe(periods=12, freq='MS')

# Forecast generate
forecast = model.predict(future)

# -------------------------------
# 3️ Future Table (Next 12 Months)
# -------------------------------

future_table = forecast[['ds', 'yhat']].tail(12).copy()

# Revenue ko round kar rahe hain
future_table['yhat'] = future_table['yhat'].round(2)

# Date ko readable format me convert
future_table['ds'] = future_table['ds'].dt.strftime('%B %Y')

# Column names clean
future_table.columns = ['Month', 'Predicted Revenue']

print("\n--- Agle 12 Mahine ka Forecast ---")
print(future_table.to_string(index=False))

# -------------------------------
# 4️ ALL GRAPHS IN ONE PAGE (2x2)
# -------------------------------

fig, ax = plt.subplots(2, 2, figsize=(14,10))

# -------------------------------
# Graph 1️ Actual vs Forecast
# -------------------------------
ax[0,0].plot(forecast['ds'], forecast['yhat'], label='Forecast')
ax[0,0].plot(df['ds'], df['y'], label='Actual Revenue')
ax[0,0].set_title("Actual vs Forecast Revenue")
ax[0,0].legend()

# -------------------------------
# Graph 2️ Trend Component
# -------------------------------
ax[0,1].plot(forecast['ds'], forecast['trend'])
ax[0,1].set_title("Trend Component")

# -------------------------------
# Graph 3️ Seasonality (Yearly)
# -------------------------------
ax[1,0].plot(forecast['ds'], forecast['yearly'])
ax[1,0].set_title("Yearly Seasonality")

# -------------------------------
# Graph 4️ Future Revenue Bar Graph
# -------------------------------
bars = ax[1,1].bar(
    future_table["Month"],
    future_table["Predicted Revenue"],
    color="steelblue"
)

# Har bar ke upar value likhna
for bar in bars:
    height = bar.get_height()
    ax[1,1].text(
        bar.get_x() + bar.get_width()/2,
        height,
        f'{int(height)}',
        ha='center',
        va='bottom',
        fontsize=8 
    )

ax[1,1].set_title("Future Monthly Revenue")
ax[1,1].tick_params(axis='x', rotation=45)

# Layout fix (overlap avoid karne ke liye)
plt.tight_layout()
plt.show()

# -------------------------------
# 5️ Future Table CSV me save
# -------------------------------
future_table.to_csv('future_revenue_forecast.csv', index=False)

# -------------------------------
# 6️ Extra Data (Future Analysis)
# -------------------------------
fc = pd.read_csv('financial_customers.csv')
