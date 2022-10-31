import streamlit as st
import pickle

model = pickle.load(open('naive_bayes_1.0.pkl','rb'))

st.write('# Fake News Detection Using Machine Learning')
text = st.text_input('Enter an article headline: ')
button  = st.button('Predict')

if button:
    prediction = model.predict(text)
    print(prediction)
    st.write(prediction)
