import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pickle

model = pickle.load(open('SleepAnalysis.pkl', 'rb'))
st.markdown("<h1 style = 'text-align: center; color: 3D0C11'>SLEEP_ANALYSIS PROJECT </h1> ", unsafe_allow_html = True)
st.markdown("<h6 style = 'top_margin: 0rem; text-align: center; color: #FFB4B4'>Built by Oluwatosin</h6>", unsafe_allow_html = True)
st.image('pngwing.com.png', width = 400)


st.subheader('Project Brief')

st.markdown("<p style = 'top_margin: 0rem; text-align: justify; color: #FFB4B4'> Analyzing and understanding sleep patterns is essential for evaluating sleep quality and identifying potential sleep-related issues. Sleep patterns typically refer to the various stages and cycles of sleep that individuals go through during a typical night's rest..</p>", unsafe_allow_html = True)

st.markdown("<br><br>", unsafe_allow_html = True)

username = st.text_input('Enter your name')
if st.button('submit name'):
    st.success(f"Welcome {username}. Pls use according to usage guidelines")

data = pd.read_csv('Sleep_Analysis.csv')
heat = plt.figure(figsize = (14, 7))
sel_col = ['Age', 'Gender', 'exercise', 'bluelight filter', 'physical illness', 'smoke/drink']
data = data[sel_col]

from sklearn.preprocessing import LabelEncoder, StandardScaler
def transformer(dataframe):
    lb = LabelEncoder()
    scaler = StandardScaler()
    dep = dataframe.drop('smoke/drink', axis=1)

     # scale the numerical columns
    for i in dep:# ---------------------------------------------- Iterate through the dataframe columns
        if i in dataframe.select_dtypes(include = 'number').columns: # --------- Select only the numerical columns
            dataframe[[i]] = scaler.fit_transform(dataframe[[i]]) # ------------ Scale all the numericals

    # label encode the categorical columns
    for i in dataframe.columns:  # --------------------------------------------- Iterate through the dataframe columns
        if i in dataframe.select_dtypes(include = ['object', 'category',]).columns: #-- Select all categorical columns
            dataframe[i] = lb.fit_transform(dataframe[i]) # -------------------- Label encode selected categorical columns
    return dataframe

transformer(data)

sns.heatmap(data.corr(), annot = True, cmap = 'BuPu')

st.write(heat)

st.write(data.sample(10))

st.sidebar.image('pngwing.com (1).png', caption= f'Welcome {username}')

input_type = st.sidebar.selectbox('Select Your preffered Input type', ['Slider Input', 'Number Input'])

if input_type == "Slider Input":
    age = st.sidebar.slider("Age", data['Age'].min(), data['Age'].max())
    exercise = st.sidebar.slider("exercise", data['exercise'].min(), data['exercise'].max())
    #bluelightfilter = st.sidebar.slider("bluelight filter", data['bluelight filter'].min(), data['bluelight filter'].max())
else:
    age = st.sidebar.number_input("Age", data['Age'].min(), data['Age'].max())
    exercise = st.sidebar.number_input("exercise", data['exercise'].min(), data['exercise'].max())
    #bluelightfilter = st.sidebar.number_input("bluelight filter", data['bluelight filter'].min(), data['bluelight filter'].max())
    
input_variable = pd.DataFrame([{"Age":age, "exercise":exercise}])
st.write(input_variable)

pred_result, interpret = st.tabs(["Prediction Tab", "Interpretation Tab"])
with pred_result:
    if st.button('PREDICT'):

        st.markdown("<br>", unsafe_allow_html= True)
        prediction = model.predict(input_variable)
        st.write("Predicted Profit is :", prediction)
    else:
        st.write('Pls press the predict button for prediction')

with interpret:
    st.subheader('Model Interpretation')
    st.write(f"smoke/drink = {model.intercept_.round(2)} + {model.coef_[0].round(2)} Age + {model.coef_[1].round(2)} exercise")

    st.markdown("<br>", unsafe_allow_html= True)

    #st.markdown(f"- The expected Profit for a startup is {model.intercept_}")

    #st.markdown(f"- For every additional 1 dollar spent on R&D Spend, the expected profit is expected to increase by ${model.coef_[0].round(2)}  ")

    #st.markdown(f"- For every additional 1 dollar spent on Administration Expense, the expected profit is expected to decrease by ${model.coef_[1].round(2)}  ")