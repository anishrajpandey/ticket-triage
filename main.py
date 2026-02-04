import pandas as pd
data= pd.read_csv('./data/customer_support_tickets.csv')
print(data.columns) 
print(data.iloc[0])

