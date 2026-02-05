import pandas as pd
data= pd.read_csv('./data/customer_support_tickets.csv')
# print(data.columns) 
# print(data.iloc[0])
print(data["Ticket Type"].unique())
# todo clean data for ticket type to be one-word-strins


