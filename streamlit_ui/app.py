from matplotlib import colors
import pandas as pd
import streamlit as st
import requests
import traceback
import os
import base64
import gc
import statsmodels.api as sm
from requests.structures import CaseInsensitiveDict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from streamlit_option_menu import option_menu

def main():

    st.set_page_config(layout="wide")
    # st.text_input("Login", "")
    # st.text_input("Logout", "")

    # names = ['John Smith','Rebecca Briggs']
    # usernames = ['jsmith','rbriggs']
    # passwords = ['123','456']

    # hashed_passwords = stauth.Hasher(passwords).generate()

    # authenticator = stauth.Authenticate(names,usernames,hashed_passwords,'some_cookie_name','some_signature_key',cookie_expiry_days=30)

    # name, authentication_status, username = authenticator.login('Login','main')

    # if authentication_status:
    #     authenticator.logout('Logout', 'main')
    #     st.write('Welcome *%s*' % (name))
    #     st.title('Some content')
    # elif authentication_status == False:
    #     st.error('Username/password is incorrect')
    # elif authentication_status == None:
    #     st.warning('Please enter your username and password')

    # if st.session_state['authentication_status']:
    #     authenticator.logout('Logout', 'main')
    #     st.write('Welcome *%s*' % (st.session_state['name']))
    #     st.title('Some content')
    # elif st.session_state['authentication_status'] == False:
    #     st.error('Username/password is incorrect')
    # elif st.session_state['authentication_status'] == None:
    #     st.warning('Please enter your username and password')

    list = ["Home", "Login"]
    option = st.sidebar.selectbox("Menu", list)

    if option == "Home":
        st.title("Product Sales Forecast")
        st.image("./fs.jpeg")

        if st.sidebar.checkbox("Show Records"):
            
            st.subheader("Dataset")
            table = pd.read_csv("./data/table.csv")
            train_data = pd.read_csv("./data/train.csv", index_col="id", header=0, parse_dates=['date'])
            stores_data = pd.read_csv("./data/stores.csv", index_col="store_nbr", header=0)
            transactions_data = pd.read_csv("./data/transactions.csv", index_col=None, header=0, parse_dates=['date'])

            st.write(table)

            total_records = train_data.shape[0]
            first_date    = train_data.date.iloc[0]
            last_date     = train_data.date.iloc[-1]
            total_days    = (train_data.date.iloc[-1] - train_data.date.iloc[0]).days
            store_nbr_id  = stores_data.index.values # stores_data.store_nbr.unique()
            family_unique = train_data.family.unique()

            st.metric("Number of Records", "{} from {} to {}".format(total_records,first_date.to_period("D"), last_date.to_period("D") ) )
            st.write("(Total of {} days or {} months)".format(total_days,total_days//30 ))

            col5, col6 = st.columns(2)
            col5.metric("Number of Stores", "{} stores".format(len(store_nbr_id)))
            col6.metric("Number of Product Family", "{} types".format(len(family_unique) ) )

            st.metric("Number of Cities and States", "{} cities in {} states". format(len(stores_data.city.unique()), len(stores_data.state.unique()) ))

    if option == "Login":
        username = st.sidebar.text_input("Username", '')
        password = st.sidebar.text_input("Password", type="password")

        # st.title("Login")
        # st.image("/Users/vachanabelgavi/Downloads/store-sales-time-series-forecasting/streamlit/login.webp", caption=None, \
        #     width=750, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

        if st.sidebar.checkbox("Login"):
            if password == '1234':

                selected = option_menu(
                menu_title="Analytics Dashboard", 
                options=["Products", "Stores", "Sales vs Oil", "HolidayEvent", "Transactions", "ACF & PACF"],
                icons=["archive", "basket2", "device-ssd", "collection", "credit-card", "boombox"],
                default_index=0, orientation="horizontal",
                styles={"container": {"padding": "0!important", "background-color": "#fafafa"},
                 "icon": {"color": "orange", "font-size": "14px"}, 
                 "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                 "nav-link-selected": {"background-color": "bg-success p-2 text-dark bg-opacity-50"},
                 })

                if selected == "Products":
                    #st.title("Analytics Dashboard")
                    page_second()

                if selected == "Stores":
                    page_third()

                if selected == "Sales vs Oil":
                    page_fourth()  

                if selected == "HolidayEvent":
                    page_fifth()  

                if selected == "Transactions":
                    page_sixth()  

                if selected == "ACF & PACF":
                    page_seventh()  

            else:
                st.sidebar.write("Username & Password do not match.")

            
def page_first():
    data = pd.read_csv("./data/results.csv")
    train = st.cache(pd.read_csv)("./data/table.csv", nrows=100)
    st.write("Dataset: ", train)

    id = data['id']
    selectedbox = st.selectbox("Product ID", id)

    x = int(selectedbox)

    if st.button("Predict"):
        #print(type(x))
        url = "https://aiwnpwql6jfb4hixhtinb5sinu0zgkyk.lambda-url.us-east-1.on.aws/predict_sales/" + str(x)
        #print(url)
        headers = CaseInsensitiveDict()
        headers["accept"] = "application/json"
        resp = requests.post(url, headers=headers, verify=False, timeout=8000)
        #print("Response: ", resp)
        result = resp.json()
        sales = result["Sales"]

        for i in range(0, len(data)):
            if data['id'][i] == x:
                row = data.at[i, 'family']

        col1, col2 = st.columns(2)
        col1.metric("Product ID", x)
        col2.metric("Sales", sales)

        st.metric("Product Family", row)


def page_second():

    st.header("Product Family vs Sales")
    train_data = pd.read_csv("./data/train.csv", index_col="id", header=0, parse_dates=['date'])

    pf = train_data['family'].unique()
    selected_option = st.selectbox("Product Family", pf)

    if st.button("Analyze"):
        st.subheader("Sales of " + selected_option + " Family")
        train_data_1 = train_data.loc[train_data['family'] == selected_option]
        sales_grouped  = train_data_1.groupby('date').agg({'sales':'sum'}).to_period("D")

        sales_grouped['year']      = sales_grouped.index.year
        sales_grouped['quarter']   = sales_grouped.index.quarter
        sales_grouped['month']     = sales_grouped.index.month
        sales_grouped['week']      = sales_grouped.index.week
        sales_grouped['dayofweek'] = sales_grouped.index.dayofweek  # Monday=0, Sunday=6
        sales_grouped['dayofmonth']= sales_grouped.index.day  # day in month from 01 to 31
        sales_grouped['dayofyear'] = sales_grouped.index.dayofyear

        sales_smooth7  = sales_grouped.copy()
        sales_smooth30 = sales_grouped.copy()
        sales_smooth365= sales_grouped.copy()

        sales_smooth7["sales"]   = sales_smooth7.  sales.rolling(window=7,  center=True, min_periods=3 ).mean()
        sales_smooth30["sales"]  = sales_smooth30. sales.rolling(window=30, center=True, min_periods=15).mean()
        sales_smooth365["sales"] = sales_smooth365.sales.rolling(window=365,center=True, min_periods=183).mean()

        figsize = (15,5)
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=figsize)
        sales_grouped.groupby(['month']).agg({'sales':'mean'}).plot(kind="barh",ax=ax1, color='#5da68d')
        ax1.set(title="Average Sales by Month")
        ax1.set(ylabel="Month", xlabel="Average Sales")
        ax1.get_legend().remove()
        labels1 = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        ax1.set_yticks(range(0,12), labels1)

        sales_grouped.groupby(['quarter']).agg({'sales':'mean'}).plot.pie(y="sales",ax=ax2, legend=False, autopct='%1.f%%',
                                    startangle=90, labels=["Quarter 1","Quarter 2","Quarter 3","Quarter 4"], 
                                    fontsize="x-large", colors=['#d2fbd4', '#a5dbc2', '#7bbcb0', '#559c9e'])
        ax2.set(title="Average Sales by Quarter")
        plt.savefig("./images/avg-sales.png")
        st.image("./images/avg-sales.png")

        fig = (10,4)
        fig, ax1 = plt.subplots(1, figsize=fig)
        sales_grouped.groupby(['year']).agg({'sales':'mean'}).plot(kind="bar",ax=ax1, color='#2e9471')
        ax1.set(title="Average Sales by Year")
        ax1.set(xlabel="Year", ylabel="Average Sales")
        ax1.get_legend().remove()
        labels1 = ['2013','2014','2015','2016','2017']
        ax1.set_xticks(range(0,5), labels1)
        plt.savefig("./images/avg-sales-year.png")
        st.image("./images/avg-sales-year.png")

def page_third():

    st.header("Store Analysis")

    l = ["Promotions per Store", "Analysis with Product Family"]
    opt = st.selectbox("Select the Analysis Type", l)

    if st.checkbox("Analysis"):
        if opt == "Promotions per Store":
            images_dir = "./images"
            
            API_URL = 'https://aiwnpwql6jfb4hixhtinb5sinu0zgkyk.lambda-url.us-east-1.on.aws/'
            store_input = st.number_input(label="Store Number", min_value=1, max_value=54, step=1)
            year_input = st.text_input(label="Year")
            
            if st.button('Submit'):
            
                URL = API_URL + f"store/{store_input}/year/{year_input}/promotions"
                r = requests.get(URL)

                try:
                    r_json = r.json()
                    if r_json.get('message') and not r_json.get('data'):
                        st.write(r_json.get('message'))
                    elif r_json:
                        image_b64 = r_json.get('data')
                        file_name = os.path.join(images_dir,f'promo-image-{int(datetime.now().timestamp())}.png')
                        with open(file_name, "wb") as file:
                            file.write(base64.b64decode(image_b64))

                        st.image(file_name)
                    else:
                        st.write(r_json.get('message'))
                except:
                    print(traceback.format_exc())
                    st.write(f'Internal error')

        if opt == "Analysis with Product Family":
            images_dir = "./images"
            
            API_URL = 'https://aiwnpwql6jfb4hixhtinb5sinu0zgkyk.lambda-url.us-east-1.on.aws/'
            store_input = st.number_input(label="Store Number", min_value=1, max_value=54, step=1)
            fams = pd.read_csv('./data/train.csv')
            fam = fams['family'].unique()
            family_input = st.selectbox("Family", fam)
            
            if st.button('Submit'):
            
                URL = API_URL + f"store/{store_input}/family/{family_input}/sales"
                r = requests.get(URL)

                try:
                    r_json = r.json()
                    if r_json.get('message') and not r_json.get('data'):
                        st.write(r_json.get('message'))
                    elif r_json:
                        data_dict = r_json.get('data')
                        image_year = data_dict.get('data_year')
                        image_month = data_dict.get('data_month')
                        timestamp = int(datetime.now().timestamp())
                        file_name_year = os.path.join(images_dir,f'sales-year-image-{timestamp}.png')
                        file_name_month = os.path.join(images_dir,f'sales-month-image-{timestamp}.png')
                        with open(file_name_year, "wb") as file:
                            file.write(base64.b64decode(image_year))
                        st.image(file_name_year)
                        with open(file_name_month, "wb") as file:
                            file.write(base64.b64decode(image_month))
                        st.image(file_name_month)
                    else:
                        st.write(r_json.get('message'))
                except:
                    print(traceback.format_exc())
                    st.write(f'Internal error')



def page_fourth():

    st.header("Oil Prices vs Product Family Sales")

    train = pd.read_csv("./data/train.csv")

    oil = pd.read_csv("./data/oil.csv")
    oil["date"] = pd.to_datetime(oil.date)
    train['date']=train['date'].astype('datetime64')

    pf = train['family'].unique()
    selected_option = st.selectbox("Product Family", pf)

    # Resample
    oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()
    # Interpolate
    oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])
    oil["dcoilwtico_interpolated"] =oil.dcoilwtico.interpolate()

    oil2 = pd.merge(train.groupby(["date", "family"]).sales.sum().reset_index(), oil.drop("dcoilwtico", axis = 1), how = "left")
    oil2_corr = oil2.groupby("family").corr("spearman").reset_index()
    oil2_corr = oil2_corr[oil2_corr.level_1 == "dcoilwtico_interpolated"][["family", "sales"]].sort_values("sales")

    if st.button("Analyze"):
        st.subheader(selected_option)
        oil2[oil2.family == selected_option].plot.scatter(x = "dcoilwtico_interpolated", y = "sales")
        plt.title(selected_option+"\n Correlation:"+str(oil2_corr[oil2_corr.family == selected_option].sales), fontsize = 12)
        plt.axvline(x=70, color='r', linestyle='--')
        plt.savefig("./images/oil-sales.png")
        st.image("./images/oil-sales.png")



def page_fifth():

    st.header("Holidays & Events Data Analysis")
    st.write("")

    # filter holidays for the training dataset window.
    holidays_events = pd.read_csv('./data/holidays_events.csv')
    train = pd.read_csv('./data/train.csv')

    holidays_events = holidays_events[(holidays_events['date'] >= "2013-01-01") & (holidays_events['date'] <= "2017-08-15")]

    ##Let's look at the sales behavior for the whole data
    train_aux = train[['date', 'sales']].groupby('date').mean()
    train_aux = train_aux.reset_index()
    fig = go.Figure(data=go.Scatter(x=train_aux['date'],
    y=train_aux['sales'],
    marker_color='red', text="sales"))

    if st.checkbox("Average Sales by Date with Holidays & Events"):
        st.write("")
        for holiday_date in list(holidays_events['date']):
            fig.add_vline(x=holiday_date, line_width=0.5, line_dash="dash", line_color="green")

        #fig.add_vline(x="2013-08-08", line_width=0.5, line_dash="dash", line_color="green", annotation="test")
        fig.update_layout({"title": f'Avg Sales by date with Holidays Events',
        "xaxis": {"title":"Date"},
        "yaxis": {"title":"Avg Unit Sold"},
        "showlegend": False})
        fig.write_image('./images/sales-holiday.png')
        st.image('./images/sales-holiday.png')
        fig.show()

    if st.checkbox("Average Sales on Holidays & Events"):

        df_plot = pd.merge(holidays_events, train_aux, on='date', how='inner')
        df_plot.loc[df_plot['description'].isin(['Black Friday', 'Cyber Monday']), 'type'] = 'black_friday_cyber_monday'

        fig = px.scatter(df_plot, x="date", y="sales", size='sales', color='type')
        fig.update_layout({"title": f'Avg Sales on Holiday Events days',
        "xaxis": {"title":"HOLIDAY EVENT DATE"},
        "yaxis": {"title":"Avg Sales"},
        "showlegend": True})

        fig.add_annotation(x='2014-07-05',y=500,xref="x",yref="y",text="WORLD CUP",showarrow=True, align="center",arrowhead=2,arrowsize=1,
        arrowwidth=2,arrowcolor="#636363",ax=0,ay=-30,bordercolor="#c7c7c7",borderwidth=2,borderpad=4,bgcolor="#ca8ee8",opacity=0.8 )

        fig.add_annotation(x='2016-04-20',y=800,xref="x",yref="y",text="EARTHQUAKE",showarrow=True,align="center",arrowhead=2,arrowsize=1,
        arrowwidth=2,arrowcolor="#636363",ax=0,ay=-30,bordercolor="#c7c7c7",borderwidth=2,borderpad=4,bgcolor="#ca8ee8",opacity=0.8)
        
        fig.add_annotation(x='2013-12-30',y=200,xref="x",yref="y",text="CHRISTAMS 13/14",showarrow=True,align="center",arrowhead=2,arrowsize=1,
        arrowwidth=2,arrowcolor="#636363",ax=0,ay=30,bordercolor="#c7c7c7",borderwidth=2,borderpad=4,bgcolor="#3ce685",opacity=0.8)

        fig.add_annotation(x='2014-12-30',y=200,xref="x",yref="y",text="CHRISTAMS 14/15",showarrow=True,align="center",arrowhead=2,arrowsize=1,
        arrowwidth=2,arrowcolor="#636363",ax=0,ay=30,bordercolor="#c7c7c7",borderwidth=2,borderpad=4,bgcolor="#3ce685",opacity=0.8)

        fig.add_annotation(x='2015-12-30',y=200,xref="x",yref="y",text="CHRISTAMS 15/16",showarrow=True,align="center",arrowhead=2,arrowsize=1,
        arrowwidth=2,arrowcolor="#636363",ax=0,ay=30,bordercolor="#c7c7c7",borderwidth=2,borderpad=4,bgcolor="#3ce685",opacity=0.8)
        
        fig.add_annotation(x='2016-12-30',y=200,xref="x",yref="y",text="CHRISTAMS 16/17",showarrow=True,align="center",arrowhead=2,arrowsize=1,
        arrowwidth=2,arrowcolor="#636363",ax=0,ay=30,bordercolor="#c7c7c7",borderwidth=2,borderpad=4,bgcolor="#3ce685",opacity=0.8)
        fig.write_image("./images/events-sales.png")
        st.image("./images/events-sales.png")
        fig.show()


def page_sixth():

    st.header("Store vs Transactions Analysis")

    selection = st.number_input("Store Number", min_value=1, max_value=54, step=1)
    if st.button("Select"):
        # Import
        train = pd.read_csv("./data/train.csv")
        test = pd.read_csv("./data/test.csv")
        stores = pd.read_csv("./data/stores.csv")
        transactions = pd.read_csv("./data/transactions.csv").sort_values(["store_nbr", "date"])

        # Datetime
        train["date"] = pd.to_datetime(train.date)
        test["date"] = pd.to_datetime(test.date)
        transactions["date"] = pd.to_datetime(transactions.date)

        # Data types
        train.onpromotion = train.onpromotion.astype("float16")
        train.sales = train.sales.astype("float32")
        stores.cluster = stores.cluster.astype("int8")

        train_data_1 = train.loc[train['store_nbr'] == selection]
        t1 = transactions.loc[transactions['store_nbr'] == selection]
        t = t1.sort_values(["date"])
        
        temp1 = pd.merge(train_data_1.groupby(["date"]).sales.sum().reset_index(), transactions, how = "left")
        #st.write("Spearman Correlation between Total Sales and Transactions: {:,.4f}".format(temp1.corr("spearman").sales.loc["transactions"]))
        figure = px.line(t, x='date', y='transactions', title = "Transactions")
        figure.write_image('./images/trans.png')
        st.image("./images/trans.png")
        figure.show()


def page_seventh():
    #%%
    holidays = pd.read_csv("./data/holidays_events.csv")
    holidays["date"] = pd.to_datetime(holidays.date)
    # Transferred Holidays
    tr1 = holidays[(holidays.type == "Holiday") & (holidays.transferred == True)].drop("transferred", axis = 1).reset_index(drop = True)
    tr2 = holidays[(holidays.type == "Transfer")].drop("transferred", axis = 1).reset_index(drop = True)
    tr = pd.concat([tr1,tr2], axis = 1)
    tr = tr.iloc[:, [5,1,2,3,4]]

    holidays = holidays[(holidays.transferred == False) & (holidays.type != "Transfer")].drop("transferred", axis = 1)
    holidays = holidays.append(tr).reset_index(drop = True)


    # Additional Holidays
    holidays["description"] = holidays["description"].str.replace("-", "").str.replace("+", "").str.replace('\d+', '')
    holidays["type"] = np.where(holidays["type"] == "Additional", "Holiday", holidays["type"])

    # Bridge Holidays
    holidays["description"] = holidays["description"].str.replace("Puente ", "")
    holidays["type"] = np.where(holidays["type"] == "Bridge", "Holiday", holidays["type"])

    
    # Work Day Holidays, that is meant to payback the Bridge.
    work_day = holidays[holidays.type == "Work Day"]  
    holidays = holidays[holidays.type != "Work Day"]

    # Split
    # Events are national
    events = holidays[holidays.type == "Event"].drop(["type", "locale", "locale_name"], axis = 1).rename({"description":"events"}, axis = 1)

    holidays = holidays[holidays.type != "Event"].drop("type", axis = 1)
    regional = holidays[holidays.locale == "Regional"].rename({"locale_name":"state", "description":"holiday_regional"}, axis = 1).drop("locale", axis = 1).drop_duplicates()
    national = holidays[holidays.locale == "National"].rename({"description":"holiday_national"}, axis = 1).drop(["locale", "locale_name"], axis = 1).drop_duplicates()
    local = holidays[holidays.locale == "Local"].rename({"description":"holiday_local", "locale_name":"city"}, axis = 1).drop("locale", axis = 1).drop_duplicates()

    #%%
    test = pd.read_csv("./data/test.csv")
    stores = pd.read_csv("./data/stores.csv")
    train = pd.read_csv("./data/train.csv")
    d = pd.merge(train, stores)
    d["store_nbr"] = d["store_nbr"].astype("int8")

    d['date']=d['date'].astype('datetime64')
    # National Holidays & Events
    #d = pd.merge(d, events, how = "left")
    d = pd.merge(d, national, how = "left")
    # Regional
    d = pd.merge(d, regional, how = "left", on = ["date", "state"])
    # Local
    d = pd.merge(d, local, how = "left", on = ["date", "city"])

    # Work Day: It will be removed when real work day column is created
    d = pd.merge(d,  work_day[["date", "type"]].rename({"type":"IsWorkDay"}, axis = 1),how = "left")

    # EVENTS
    events["events"] =np.where(events.events.str.contains("futbol"), "Futbol", events.events)
    events, events_cat = one_hot_encoder(events, nan_as_category=False)
    events["events_Dia_de_la_Madre"] = np.where(events.date == "2016-05-08", 1,events["events_Dia_de_la_Madre"])
    events = events.drop(239)

    d = pd.merge(d, events, how = "left")
    d[events_cat] = d[events_cat].fillna(0)

    #%%
    # New features
    d["holiday_national_binary"] = np.where(d.holiday_national.notnull(), 1, 0)
    d["holiday_local_binary"] = np.where(d.holiday_local.notnull(), 1, 0)
    d["holiday_regional_binary"] = np.where(d.holiday_regional.notnull(), 1, 0)

    # 
    d["national_independence"] = np.where(d.holiday_national.isin(['Batalla de Pichincha',  'Independencia de Cuenca', 'Independencia de Guayaquil', 'Independencia de Guayaquil', 'Primer Grito de Independencia']), 1, 0)
    d["local_cantonizacio"] = np.where(d.holiday_local.str.contains("Cantonizacio"), 1, 0)
    d["local_fundacion"] = np.where(d.holiday_local.str.contains("Fundacion"), 1, 0)
    d["local_independencia"] = np.where(d.holiday_local.str.contains("Independencia"), 1, 0)


    holidays, holidays_cat = one_hot_encoder(d[["holiday_national","holiday_regional","holiday_local"]], nan_as_category=False)
    d = pd.concat([d.drop(["holiday_national","holiday_regional","holiday_local"], axis = 1),holidays], axis = 1)

    he_cols = d.columns[d.columns.str.startswith("events")].tolist() + d.columns[d.columns.str.startswith("holiday")].tolist() + d.columns[d.columns.str.startswith("national")].tolist()+ d.columns[d.columns.str.startswith("local")].tolist()
    d[he_cols] = d[he_cols].astype("int8")

    d[["family", "city", "state", "type"]] = d[["family", "city", "state", "type"]].astype("category")

    del holidays, holidays_cat, work_day, local, regional, national, events, events_cat, tr, tr1, tr2, he_cols
    gc.collect()

    d['date'] = pd.to_datetime(d['date'], errors='coerce')

    d = create_date_features(d)
    # Workday column
    d["workday"] = np.where((d.holiday_national_binary == 1) | (d.holiday_local_binary==1) | (d.holiday_regional_binary==1) | (d['day_of_week'].isin([6,7])), 0, 1)
    d["workday"] = pd.Series(np.where(d.IsWorkDay.notnull(), 1, d["workday"])).astype("int8")
    d.drop("IsWorkDay", axis = 1, inplace = True)

    # Wages in the public sector are paid every two weeks on the 15th and on the last day of the month. 
    # Supermarket sales could be affected by this.
    d["wageday"] = pd.Series(np.where((d['is_month_end'] == 1) | (d["day_of_month"] == 15), 1, 0)).astype("int8")

    menu = train['family'].unique()
    f = st.selectbox("Product Family", menu)

    lag = [30, 60, 90, 120, 365, 730]
    l = st.selectbox("Number of Lags", lag)

    if st.checkbox("Submit"):

        a = d[(d.sales.notnull())].groupby(["date", "family"]).sales.mean().reset_index().set_index("date")
        try:
            fig, ax = plt.subplots(1,2,figsize=(15,5))
            temp = a[(a.family == f)]
            sm.graphics.tsa.plot_acf(temp.sales, lags=l, ax=ax[0], title = "AUTOCORRELATION\n" + f)
            sm.graphics.tsa.plot_pacf(temp.sales, lags=l, ax=ax[1], title = "PARTIAL AUTOCORRELATION\n" + f)
            plt.savefig('./images/acf-pacf.png')
            st.image('./images/acf-pacf.png')
        except:
            pass



def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = df.select_dtypes(["category", "object"]).columns.tolist()
    # categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    df.columns = df.columns.str.replace(" ", "_")
    return df, df.columns.tolist()

# Time Related Features
def create_date_features(df):
    df['month'] = df.date.dt.month.astype("int8")
    df['day_of_month'] = df.date.dt.day.astype("int8")
    df['day_of_year'] = df.date.dt.dayofyear.astype("int16")
    df['week_of_month'] = (df.date.apply(lambda d: (d.day-1) // 7 + 1)).astype("int8")
    df['week_of_year'] = (df.date.dt.weekofyear).astype("int8")
    df['day_of_week'] = (df.date.dt.dayofweek + 1).astype("int8")
    df['year'] = df.date.dt.year.astype("int32")
    df["is_wknd"] = (df.date.dt.weekday // 4).astype("int8")
    df["quarter"] = df.date.dt.quarter.astype("int8")
    df['is_month_start'] = df.date.dt.is_month_start.astype("int8")
    df['is_month_end'] = df.date.dt.is_month_end.astype("int8")
    df['is_quarter_start'] = df.date.dt.is_quarter_start.astype("int8")
    df['is_quarter_end'] = df.date.dt.is_quarter_end.astype("int8")
    df['is_year_start'] = df.date.dt.is_year_start.astype("int8")
    df['is_year_end'] = df.date.dt.is_year_end.astype("int8")
    # 0: Winter - 1: Spring - 2: Summer - 3: Fall
    df["season"] = np.where(df.month.isin([12,1,2]), 0, 1)
    df["season"] = np.where(df.month.isin([6,7,8]), 2, df["season"])
    df["season"] = pd.Series(np.where(df.month.isin([9, 10, 11]), 3, df["season"])).astype("int8")
    return df


if __name__ == "__main__":
    main()