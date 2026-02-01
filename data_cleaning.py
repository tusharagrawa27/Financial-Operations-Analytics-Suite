import pandas as pd
import numpy as np

fc = pd.read_csv('financial_customers.csv')
ft = pd.read_csv('financial_transactions.csv')
mr = pd.read_csv('monthly_revenue.csv')
# we will check the info of all the three csv's so that we can found how many null values are their
print("HERE IS THE INFO OF FINANCIAL_CUSTOMERS\n")
print( fc.info()) #no null
print("HERE IS THE INFO OF FINANCIAL_TRANSACTIONS\n")
print( ft.info()) #no null
print("HERE IS THE INFO OF MONTHLY_REVENUE\n")
print( mr.info()) #no null
# we will check if there are any duplicate rows in the three of csv's
print("HERE IS THE COUNT OF FINANCIAL_CUSTOMERS DUPLICATES :",fc.duplicated().sum()) #0
print("HERE IS THE COUNT OF FINANCIAL_TRANSACTIONS DUPLICATES :",ft.duplicated().sum()) #0
print("HERE IS THE COUNT OF MONTHLY_REVENUE DUPLICATES :",mr.duplicated().sum()) #0
# WE WILL CHECK THE DESCRIBTION OF ALL THE THREE CSV'S
print("HERE IS THE DESCRIBTION OF FINANCIAL_CUSTOMERS \n")
print( fc.describe())
print("HERE IS THE DESCRIBTION OF FINANCIAL_TRANSACTIONS\n")
print( ft.describe())
print("HERE IS THE DESCRIBTION OF MONTHLY_REVENUE\n")
print( mr.describe())


# DATA IS ALREADY CLEANED 
print("\n\nTASK DONE")


