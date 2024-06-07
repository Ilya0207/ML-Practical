from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

# Load the trained model
model = joblib.load('best_svm_model.pkl')

# Define the FastAPI app
app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the prediction endpoint
@app.post("/predict")
async def predict(
    Age: Optional[int] = None,
    Month: Optional[str] = None,
    Occupation: Optional[str] = None,
    Annual_Income: Optional[float] = None,
    Monthly_Inhand_Salary: Optional[float] = None,
    Num_Bank_Accounts: Optional[int] = None,
    Num_Credit_Card: Optional[int] = None,
    Interest_Rate: Optional[int] = None,
    Delay_from_due_date: Optional[int] = None,
    Num_of_Delayed_Payment: Optional[int] = None,
    Changed_Credit_Limit: Optional[float] = None,
    Num_Credit_Inquiries: Optional[int] = None,
    Credit_Mix: Optional[str] = None,
    Outstanding_Debt: Optional[float] = None,
    Credit_Utilization_Ratio: Optional[float] = None,
    Credit_History_Age: Optional[int] = None,
    Payment_of_Min_Amount: Optional[str] = None,
    Total_EMI_per_month: Optional[float] = None,
    Amount_invested_monthly: Optional[float] = None,
    Payment_Behaviour: Optional[str] = None,
    Monthly_Balance: Optional[float] = None,
    Count_Auto_Loan: Optional[int] = None,
    Count_Credit_Builder_Loan: Optional[int] = None,
    Count_Debt_Consolidation_Loan: Optional[int] = None,
    Count_Home_Equity_Loan: Optional[int] = None,
    Count_Personal_Loan: Optional[int] = None,
    Count_Not_Specified: Optional[int] = None,
    Count_Mortgage_Loan: Optional[int] = None,
    Count_Student_Loan: Optional[int] = None,
    Count_Payday_Loan: Optional[int] = None
):
    input_data = {
        "Month": Month,
        "Age": Age,
        "Occupation": Occupation,
        "Annual_Income": Annual_Income,
        "Monthly_Inhand_Salary": Monthly_Inhand_Salary,
        "Num_Bank_Accounts": Num_Bank_Accounts,
        "Num_Credit_Card": Num_Credit_Card,
        "Interest_Rate": Interest_Rate,
        "Delay_from_due_date": Delay_from_due_date,
        "Num_of_Delayed_Payment": Num_of_Delayed_Payment,
        "Changed_Credit_Limit": Changed_Credit_Limit,
        "Num_Credit_Inquiries": Num_Credit_Inquiries,
        "Credit_Mix": Credit_Mix,
        "Outstanding_Debt": Outstanding_Debt,
        "Credit_Utilization_Ratio": Credit_Utilization_Ratio,
        "Credit_History_Age": Credit_History_Age,
        "Payment_of_Min_Amount": Payment_of_Min_Amount,
        "Total_EMI_per_month": Total_EMI_per_month,
        "Amount_invested_monthly": Amount_invested_monthly,
        "Payment_Behaviour": Payment_Behaviour,
        "Monthly_Balance": Monthly_Balance,
        "Count_Auto Loan": Count_Auto_Loan,  # Modify the parameter name here
        "Count_Credit-Builder Loan": Count_Credit_Builder_Loan,
        "Count_Personal Loan": Count_Personal_Loan,
        "Count_Home Equity Loan": Count_Home_Equity_Loan,
        "Count_Not Specified": Count_Not_Specified,
        "Count_Mortgage Loan": Count_Mortgage_Loan,
        "Count_Student Loan": Count_Student_Loan,
        "Count_Debt Consolidation Loan": Count_Debt_Consolidation_Loan,
        "Count_Payday Loan": Count_Payday_Loan
    }
    
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return {"prediction": prediction[0]}
