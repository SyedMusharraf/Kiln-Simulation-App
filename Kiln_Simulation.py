import pandas as pd 
import numpy as np
import streamlit as st
import joblib
import plotly.graph_objects as go 
import plotly.express as px 
from datetime import datetime
from sklearn.preprocessing import StandardScaler

model = joblib.load("StreamLit/Kiln_sm_model.pkl")
scaler = joblib.load("StreamLit/kiln_sm_scaler.pkl")
feature_names = joblib.load('StreamLit/kiln_sm_features.pkl')

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

st.title("Kiln Simulation App")
st.subheader('Adjust the slider and get the t_gas and t_solid Output')

input, output = st.columns([2,1]) # this means input 2/3 and output - 1/3

with input:
    st.header('The inputs Parameter')
    input_data = {}
    for feature in feature_names:
        input_data[feature]=st.slider(f"{feature}",
            max_value=150.00,
            min_value=0.00,
            step=0.10,
            value = 20.00,
            format='%.2f'
        )

    input_df = pd.DataFrame([input_data],columns=feature_names)

    input_scaled = scaler.transform(input_df)
    predictions = model.predict(input_scaled)
    st.session_state.predictions = predictions[0] # store the value 

    #add button for time series 
    if st.button('Add to Time Series',type='primary'):
        current_time = datetime.now()
        st.session_state.prediction_history.append({
            'timestamp':current_time,
            'T_Solid_output':st.session_state.predictions[0],
            'T_Gas_output':st.session_state.predictions[1],
            'inputs':input_data.copy()
        })
        st.success(f"Prediction added at {current_time.strftime('%H:%M:%S')}")


with output:
    st.header('Prediction Results')
    if 'predictions' in st.session_state:
        st.write(f'**T_Solid_output**: {st.session_state.predictions[0]:.4f}')
        st.write(f'**T_Gas_output**: {st.session_state.predictions[1]:.4f}')

    else:
        st.write(f"No prediction Yet ")

st.header('Time Series Analysis')

if len(st.session_state.prediction_history)>0:

    #onverting into data frame
    history_df = pd.DataFrame(st.session_state.prediction_history)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x = history_df['timestamp'],
        y = history_df['T_Solid_output'],
        mode = 'lines+markers',
        name = 'T_Solid_output',
        line = dict(color='blue',width=2),
        marker = dict(size=8)
    ))


    fig.add_trace(go.Scatter(
        x = history_df['timestamp'],
        y = history_df['T_Gas_output'],
        mode ='lines+markers',
        name = "T_Gas_output",
        line = dict(color='red',width=2),
        marker = dict(size=8)
    ))

    fig.update_layout(
        title = 'Kiln Prediction Over Time',
        xaxis_title = 'Time',
        yaxis_title = 'Temperature Output',
        hovermode = 'x unified',
        showlegend = True,
        height = 500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction History")
    display_df = history_df.copy()
    display_df['Time']= display_df['timestamp'].dt.strftime('%H:%M:%S')
    display_df = display_df[['Time','T_Solid_output','T_Gas_output']].round(4)
    st.dataframe(display_df, use_container_width=True)

    if st.button('Clear History' , type='secondary'):
        st.session_state.prediction_history = []
        st.rerun()
else:  
    st.info("No prediction history yet ")




