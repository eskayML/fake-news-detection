import streamlit as st
import pickle

model = pickle.load(open('naive_bayes_1.0.pkl','rb'))

st.write('# Fake News Detection Using Machine Learning')
text = st.text_input('Enter an article headline into the prompt and the model will decide based on its training data , whether the news is real or not : ')
button  = st.button('Predict')

if button:
    prediction = model.predict([text])[0]
    color = 'red' if prediction =='FAKE' else 'green'
    st.write('OutCome of Prediction')
    st.write(f"## {prediction} ")
