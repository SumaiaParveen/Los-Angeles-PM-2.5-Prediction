import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.title("Air Quality Index of Los Angeles, CA")


st.sidebar.header('About the App')
st.sidebar.header('  ')

# Images
from PIL import Image
img = Image.open("la.jpg")
st.sidebar.image(img, width=300, caption="Photo: USA Today. Air Pollution of Los Angeles")

st.sidebar.markdown("### This application has been built based on a machine learning algorithm. All the input features have to be inserted in correct units (mentioned within parentheses beside each predictor) to get a result. The main pollutant of Los Angeles air is PM 2.5 and the objective of this app is to predict the concentration of PM 2.5 in micro-gram/cubic-meter. The accuracy of the model is 82.62%.")

   
avg_temp = st.number_input('Average Temperature (°C)')
max_temp = st.number_input('Maximum Temperature (°C)')
min_temp = st.number_input('Minimum Temperature (°C)')
sealevel_pressure = st.number_input('Atmospheric Sea Level Pressure (hPa)')
avg_humidity = st.number_input('Average Relative Humidity (%)')
rainfall_snowmelt = st.number_input('Total Rainfall and/or Snowmelt (mm)')
visibility = st.number_input('Average Visibility (km)')
avg_windspeed = st.number_input('Average Wind Speed (km/h)')
max_windspeed = st.number_input('Maximum Sustained Wind Speed (km/h)')

feat = [avg_temp, max_temp, min_temp, sealevel_pressure, avg_humidity, rainfall_snowmelt, visibility, avg_windspeed, max_windspeed]

data = {'avg_temp': avg_temp, 
            'max_temp': max_temp, 
            'min_temp': min_temp, 
            'sealevel_pressure': sealevel_pressure, 
            'avg_humidity': avg_humidity, 
            'rainfall_snowmelt': rainfall_snowmelt, 
            'visibility': visibility, 
            'avg_windspeed': avg_windspeed, 
            'max_windspeed': max_windspeed} 
    
df = pd.DataFrame(data, index=[0])


st.subheader('User Input parameters')
st.write(df)

feat = [float(x) for x in feat]
final_features = [np.array(feat)]

model = pickle.load(open('RF_regression_LA.pkl', 'rb'))  # get the model

st.subheader('   ')

if st.button('Predict'):
    prediction = model.predict(final_features)
    st.success(f'Concentration of PM 2.5 is : {round(prediction[0], 4)} micro-gram/cubic-meter')






