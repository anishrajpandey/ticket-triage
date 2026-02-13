import pandas as pd
from sklearn.model_selection import train_test_split

data= pd.read_csv('./data/customer_support_tickets.csv')



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


    data.rename(columns={"Ticket Type":"label",}, inplace=True)
    data["label"]= data["label"].map(ticket_type_map)   # more beautification 
    data["text"] = (
        "SUBJECT: " + data["Ticket Subject"].fillna("") +
        " DESCRIPTION: " + data["Ticket Description"].fillna("")
    )
    data.drop(columns=["Ticket Subject", "Ticket Description"], inplace=True) # remove subject and description columns



    # remove placeholders

    data["text"] = data["text"].str.replace(
        r"\{.*?\}", "", regex=True
    )

    # removing the greeting and closing statements
    boilerplate = [
    "I'm having an issue with",
    "Please assist",
    "Thank you",
    "Thanks"
    ]

    for phrase in boilerplate:
        data["text"] = data["text"].str.replace(
            phrase, "", regex=False
        )
    data["text"] = data["text"].str.replace("\n", " ", regex=False)
    data.to_csv("./data/preprocessed_tickets.csv", index=False)

  

    

preProcessData(data)
x= data["text"]
y= data["label"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
# sdds