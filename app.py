import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#from pandas_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
import pickle 

#import model 
naive_bayes = pickle.load(open('NaiveBayes.pkl','rb'))

#load dataset
data = pd.read_csv('Bank Customer Churn Dataset.csv')
#data = data.drop(data.columns[0],axis=1)

st.title('Aplikasi Bank Customer')

html_layout1 = """
<br>
<div style="background-color:blue ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Bank Customer Churn</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['Naive Bayes','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Customer Bank')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset customer</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

#if st.checkbox('EDa'):
    #pr =ProfileReport(data,explorative=True)
    #st.header('**Input Dataframe**')
    #st.write(data)
    #st.write('---')
    #st.header('**Profiling Report**')
    #st_profile_report(pr)

le=LabelEncoder()
data["country"]=le.fit_transform(data["country"])
data["gender"]=le.fit_transform(data["gender"])

#train test split
x = data.drop('churn',axis=1)
y = data['churn']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)


#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(x_train.head())
    st.write(x_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(x_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    customer_id = st.sidebar.slider('customer_id',0,1600000,1)
    credit_score = st.sidebar.slider('credit_score',0,900,600)
    country = st.sidebar.slider('country',0,4,1)
    gender = st.sidebar.slider('gender',0,2,1)
    age = st.sidebar.slider('age',0,100,59)
    tenure = st.sidebar.slider('tenure',0,15,2)
    balance = st.sidebar.slider('balance',0,2000,120)
    products_number = st.sidebar.slider('products_number',0,10,5)
    credit_card = st.sidebar.slider('credit_card', 0.05,2.5,0.45)
    active_member = st.sidebar.slider('active_member',0,10,1)
    estimated_salary = st.sidebar.slider('estimated_salary',21,100,24)
    
    user_report_data = {
        'customer_id':customer_id,
        'credit_score':credit_score,
        'country':country,
        'gender':gender,
        'age':age,
        'tenure':tenure,
        'balance':balance,
        'products_number':products_number,
        'credit_card':credit_card,
        'active_member':active_member,
        'estimated_salary':estimated_salary
        
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Nasabah
user_data = user_report()
st.subheader('Data Nasabah')
st.write(user_data)

user_result = naive_bayes.predict(user_data)
naive_bayes_score = accuracy_score(y_test,naive_bayes.predict(x_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Kamu masih menjadi nasabah aktif di bank ini'
else:
    output ='Kamu sudah berhenti menjadi nasabah bank ini'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(naive_bayes_score*100)+'%')



