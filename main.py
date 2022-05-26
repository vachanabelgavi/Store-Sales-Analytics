from helper import download_from_s3
import matplotlib.pyplot as plt 
import json
import pandas as pd
from mangum import Mangum
from fastapi import FastAPI
import uvicorn
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from datetime import datetime
import base64
import os
import traceback

###############################################################################################

app = FastAPI(
    title = "NEW API",
    description = "API helps predict product sales"
)

IMAGES_PATH = "/tmp/images/"
if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)


store_grouped = pd.read_csv("./data/store_grouped.csv")
train = pd.read_csv("./data/train.csv")

@app.get("/")
def read_root():     
    return {"message": "Welcome from the Store Sales Application"}

user_input = 3000934

@app.post("/predict")
def sales_predict():
    holidays_events = pd.read_csv(
        "./data/holidays_events.csv",
        dtype={
            'type': 'category',
            'locale': 'category',
            'locale_name': 'category',
            'description': 'category',
            'transferred': 'bool',
        },
        parse_dates=['date'],
        infer_datetime_format=True,
    )
    holidays_events = holidays_events.set_index('date').to_period('D')

    store_sales = pd.read_csv(
        './data/train.csv',
        usecols=['store_nbr', 'family', 'date', 'sales'],
        dtype={
            'store_nbr': 'category',
            'family': 'category',
            'sales': 'float32',
        },
        parse_dates=['date'],
        infer_datetime_format=True,
    )
    store_sales['date'] = store_sales.date.dt.to_period('D')
    store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()
    average_sales = (
        store_sales
        .groupby('date').mean()
        .squeeze()
        .loc['2017']
    )

    ###############################################################################################

    y = store_sales.unstack(['store_nbr', 'family']).loc["2017"] 

    # Create training data
    fourier = CalendarFourier(freq='M', order=4)
    dp = DeterministicProcess(
        index=y.index,
        constant=True,
        order=1,
        seasonal=True,
        additional_terms=[fourier],
        drop=True,
    )
    X = dp.in_sample()
    X['NewYear'] = (X.index.dayofyear == 1)

    #model = Ridge(alpha=0.5, fit_intercept=False)
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    y_pred = pd.DataFrame(model.predict(X), index=X.index, columns=y.columns)

    #################################################################################################

    df_test = pd.read_csv(
        './data/test.csv',
        dtype={
            'store_nbr': 'category',
            'family': 'category',
            'onpromotion': 'uint32',
        },
        parse_dates=['date'],
        infer_datetime_format=True,
    )
    df_test['date'] = df_test.date.dt.to_period('D')
    df_test = df_test.set_index(['store_nbr', 'family', 'date']).sort_index()

    # Create features for test set
    X_test = dp.out_of_sample(steps=16)
    X_test.index.name = 'date'
    X_test['NewYear'] = (X_test.index.dayofyear == 1)

    y_submit = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)
    y_submit = y_submit.stack(['store_nbr', 'family'])
    y_submit = y_submit.join(df_test.id).reindex(columns=['id', 'sales'])
    y_submit.to_csv('./data/submission.csv', index=False)
    return {"Submission":y_submit}

###############################################################################################

@app.post("/predict_sales/{product_id}")
def sales_prediction(product_id: int):
    df = pd.read_csv("./data/results.csv")

    for i in range(0, len(df)):
        if df['id'][i] == product_id:
            row = df.at[i, 'sales']
    return {"Sales":row}

###############################################################################################

@app.get("/store/{store_number}/year/{year}/promotions")
def store_promotion(store_number: int, year: int):
    try:
        promotion_details = store_grouped.loc[(store_grouped['store_nbr'] == store_number) & (store_grouped['year'] == year)]
        #promotion_json = promotion_details.to_json()
        promotion_dict = promotion_details.to_dict('records')

        promotion_details.plot(x="family", y="onpromotion_consolidated", kind="bar", title=f"Store - {store_number}, Year - {year}")
        file_name = os.path.join(IMAGES_PATH, f"image-{int(datetime.now().timestamp())}.png")
        plt.savefig(file_name)
        with open(file_name, "rb") as file:
            image_bytes: bytes = base64.b64encode(file.read())
        return {"data": image_bytes, "message": None}

    except Exception as e:
        print("An internal server error occured")
        print(traceback.format_exc())
        return {"data": None, "message": "An internal server error occured"}

################################################################################################

@app.get("/store/{store_number}/family/{family}/sales")
def store_sales(store_number: int, family: str):
    try:
        train_data = train.loc[(train['store_nbr'] == store_number) & (train['family'] == family)]
        train_data['date'] = pd.to_datetime(train_data['date'],format='%Y-%m-%d')
        train_data['month'] = pd.DatetimeIndex(train_data['date']).month
        train_data['year'] = pd.DatetimeIndex(train_data['date']).year
        sales_grouped_month = train_data.groupby('month').agg({'sales':'sum'})
        sales_grouped_year = train_data.groupby('year').agg({'sales':'sum'})

        timestamp = int(datetime.now().timestamp())
        file_name_year = os.path.join(IMAGES_PATH, f"sales-year-image-{timestamp}.png")
        file_name_month = os.path.join(IMAGES_PATH, f"sales-month-image-{timestamp}.png")
        sales_grouped_year.plot(kind='bar', title=f"Store - {store_number}, Family - {family}")
        plt.savefig(file_name_year)
        with open(file_name_year, "rb") as file:
            image_bytes_year: bytes = base64.b64encode(file.read())
        sales_grouped_month.plot(kind='bar', title=f"Store - {store_number}, Family - {family}")
        plt.savefig(file_name_month)
        with open(file_name_month, "rb") as file:
            image_bytes_month: bytes = base64.b64encode(file.read())
        return {"data": {"data_year": image_bytes_year, "data_month": image_bytes_month}, "message": None}
    
    except Exception as e:
        print("An internal server error occured")
        print(traceback.format_exc())
        return {"data": None, "message": "An internal server error occured"}
    

##########################################################################################
    
handler = Mangum(app)
