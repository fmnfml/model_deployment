#prediction code

import streamlit as st
import joblib
import numpy as np

#Memuat model machine learning
model = joblib.load('best_model.pkl')

def main():
    st.title('Churn Prediction')

    #Komponen input user untuk 3 fitur
    Age = st.slider('Age', min_value=18, max_value=60, value=18)
    NumOfProducts = st.slider('NumOfProducts', min_value=1, max_value=4, value=1)
    IsActiveMember = st.radio('IsActiveMember', [0, 1], index=1)
    
    if st.button('Make Prediction'):
            features = [Age, NumOfProducts, IsActiveMember]   
            result = make_prediction(features)
            if result == 1:
                st.error('Churn')
            else:
                st.success('Not Churn')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()

