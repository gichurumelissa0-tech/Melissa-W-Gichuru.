import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
# path set-up
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "data", "Tv DataSet.csv")
model_dir = os.path.join(base_dir, "models")

os.makedirs(model_dir, exist_ok=True)
# load data set
data = pd.read_csv( data_path )
#strip string
for c in ['Day_of_the_week','Show_name','Type_of_show','Show_popularity']:
    if c in data.columns:
        data[c]= data[c].astype(str).str.strip()
#parse time
data['Start_time']= pd.to_datetime(data['Start_time'], format="%I:%M%p", errors='coerce')
data['End_time']= pd.to_datetime(data['End_time'], format="%I:%M%p", errors='coerce')        
data.loc[data['End_time'] < data['Start_time'],'End_time']+= pd.Timedelta(days=1)
#hours & durations
data['Start_hour'] = data['Start_time'].dt.hour + data['Start_time'].dt.minute / 60
data['End_hour'] = data['End_time'].dt.hour + data['End_time'].dt.minute / 60
data['Duration'] = ((data['End_time'] - data['Start_time']).dt.total_seconds() / 60).fillna(0)
#cynical feature
data['Start_sin'] = np.sin(2*np.pi*data['Start_hour']/24)
data['Start_cos'] = np.cos(2*np.pi*data['Start_hour']/24)
#popularity numeric
pop_map= {'Very Low':0, 'Low':1, 'Moderate':2, 'High':3, 'Very High':4}
data['Popularity_num'] = data['Show_popularity'].map(pop_map).fillna(0).astype(int)
#encode categorical
le_show = LabelEncoder()
data['Show_label'] = le_show.fit_transform(data['Show_name'])

le_type = LabelEncoder()
data['Type_label'] = le_type.fit_transform(data['Type_of_show'])

day_map = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
data['Day_num'] = data['Day_of_the_week'].map(day_map).fillna(0).astype(int)
data['Day_sin'] = np.sin(2 * np.pi * data['Day_num'] / 7)
data['Day_cos'] = np.cos(2 * np.pi * data['Day_num'] / 7)
#save encoders
encoders = {
        'le_show': le_show,
        'le_type': le_type,
        'pop_map': pop_map,
        'day_map': day_map
}
joblib.dump(encoders, os.path.join(model_dir,'encoders.joblib'))
#features
features= ['Show_label','Type_label','Duration','Popularity_num','Start_sin','Start_cos','Day_num','Day_sin','Day_cos']
x= data[features].fillna(0)
y_viewers= data['Viewers'].astype(float)
#train/test split
x_train, x_test, y_train, y_test = train_test_split(x,y_viewers, test_size=0.3, random_state=42)
#linear regression
view_lr = LinearRegression()
view_lr.fit(x_train, y_train)
view_lr_pred= view_lr.predict(x_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, view_lr_pred))
r2_lr = r2_score(y_test, view_lr_pred)
print("Viewers LR RMSE:", rmse_lr)
print("Viewers LR R2:", r2_lr)
#randomforest
view_rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
view_rf.fit(x_train, y_train)
view_rf_pred = view_rf.predict(x_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, view_rf_pred))
r2_rf = r2_score(y_test, view_rf_pred)
print("Viewers RF RMSE:", rmse_rf)
print("Viewers RF R2:", r2_rf)
#save viewer model
joblib.dump(view_rf, os.path.join(model_dir, 'rf_viewers.joblib'))
joblib.dump(view_lr, os.path.join(model_dir, 'lr_viewers.joblib'))
print("Models saved successfully in", model_dir)

#ad revenue model
data_features= data.copy()
data_features['Predicted_viewers']= view_rf.predict(data[features].fillna(0))
features_rev= features+['Predicted_viewers']
xr= data_features[features_rev].fillna(0)
y_revenue= data_features['Ad_revenue'].astype(float)
#train/test split
xr_train, xr_test, yr_train, yr_test= train_test_split(xr, y_revenue,test_size=0.2, random_state=42)
#randomforest
rev_rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rev_rf.fit(xr_train, yr_train)
rev_rf_pred = rev_rf.predict(xr_test)

rmse_rev = np.sqrt(mean_squared_error(yr_test, rev_rf_pred))
r2_rev = r2_score(yr_test, rev_rf_pred)
print("Revenue RF RMSE:", rmse_rev)
print("Revenue RF R2:", r2_rev)
#save ad revenue model
joblib.dump(rev_rf, os.path.join(model_dir, 'rf_revenue.joblib'))
print("Models saved successfully in", model_dir)
