import pandas as pd
data= pd.read_csv('./data/customer_support_tickets.csv')
# data pre-processing

def preProcessData(data):
    data=data[["Ticket Subject",
    "Ticket Description",
    "Ticket Type"]] # remove irrelevant columns


    ticket_type_map = {
    "Refund request": "refund",
    "Technical issue": "technical",
    "Cancellation request": "cancellation",
    "Product inquiry": "product",
    "Billing inquiry": "billing"
    }


    data.rename(columns={"Ticket Type":"label", "Ticket Subject":"subject", "Ticket Description":"description"}, inplace=True)
    data["label"]= data["label"].map(ticket_type_map)   # more beautification 
    data["text"]=data["subject"] +" " + data["description"]
    data.drop(columns=["subject", "description"], inplace=True) # remove subject and description columns
    print(data.head())
  




if __name__=="__main__":
    preProcessData(data)
