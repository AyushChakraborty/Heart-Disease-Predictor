from tensorflow import keras
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image

heart_model = keras.models.load_model(r"C:\Users\chakr\OneDrive\Desktop\AI_bootcamp\Project\saved_model1")

st.set_page_config(page_title='Heart Disease Predictor', page_icon=':heart:', layout='wide')

st.title('Heart Disease Predictor :heart:')

st.subheader('This site is used to predict if CVDs or Cardiovascular Diseases are present in a person or not. CVDs are a set of disorders in the heart and blood vessels which often leads to heart failiures, strokes etc. CVDs are one of the major causes of deaths in the world and early detection of this is very important so that medication and counselling can be given early on, enabling people to live a better and healthy life')

st.subheader('Please enter the following values to proceed')

left_col, right_col = st.columns(2)

with left_col:
    age = st.text_input("Age")

    sex = st.text_input("Sex (Enter 0 for M, 1 for F)")

    chestpaintype = st.text_input("Chest Pain Type (Enter 0 for Typical Angina, 1 for Atypical Angina, 2 for Non-Anginal Pain, 3 for Asymptomatic")

    RestingBP = st.text_input("Resting BP (in mm Hg)")

    cholesterol = st.text_input("Cholesterol (Serum Cholesterol in mm/dl)")

    fastingbs = st.text_input("Fasting Blood Sugar (Enter 1 if it is greater than 120mg/dl else enter 0)")

with right_col:
    restingecg = st.text_input("Resting ECG (Enter 0 if Normal, 1 if ST, 2 if LVG)")

    maxhr = st.text_input("Maximum heart rate")

    exerciseAngina = st.text_input("Exercise Induced Angina (0 if Yes, 1 if No)")

    oldpeak = st.text_input("Old Peak (ST depression induced by exercise relative to rest)")

    STslope = st.text_input("ST Slope (Enter 0 if Up, 1 if Flat, 2 if Down)")



heart_disease = pd.read_csv(r'C:\Users\chakr\OneDrive\Desktop\AI_bootcamp\Project\heart.csv')


encode1 = {'Sex': {'M':0, 'F':1}}
encode2 = {'ChestPainType': {'TA':0, 'ATA': 1, 'NAP': 2, 'ASY':3}}
encode3 = {'RestingECG': {'Normal':0, 'ST': 1, 'LVH': 2}}
encode4 = {'ExerciseAngina': {'Y':0, 'N':1}}
encode5 = {'ST_Slope' : {'Up': 0, 'Flat': 1, 'Down': 2}}

heart_disease.replace(encode1, inplace=True)
heart_disease.replace(encode2, inplace=True)
heart_disease.replace(encode3, inplace=True)
heart_disease.replace(encode4, inplace=True)
heart_disease.replace(encode5, inplace=True)

y = heart_disease['HeartDisease']
x = heart_disease.drop(columns=['HeartDisease'])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

x_num = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
x_cat = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

df = pd.DataFrame({'Age': [float(age)], 'Sex': [int(sex)], 'ChestPainType': [int(chestpaintype)], 'RestingBP': [float(RestingBP)], 'Cholesterol': [float(cholesterol)], 
                    'FastingBS': [int(fastingbs)], 'RestingECG': [int(restingecg)], 'MaxHR': [float(maxhr)], 'ExerciseAngina': [int(exerciseAngina)], 
                    'Oldpeak': [float(oldpeak)], 'ST_Slope': [int(STslope)]})

x_test_copy = x_test.copy()

x_test_copy = x_test_copy.append(df)

 # scaling the numerical features of x_test_copy to which the new entries entered by the user are also present as a row
scaler = StandardScaler()
x_train.loc[:, ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']] = scaler.fit_transform(x_train.loc[:, ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']])
x_test_copy.loc[:, ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']] = scaler.transform(x_test_copy.loc[:, ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']])

y_pred = heart_model.predict(x_test_copy.iloc[-1:])

heart_pred = []
for i in range(0,y_pred.shape[0]):
    heart_pred.append(np.argmax(y_pred[i]))
print(heart_pred)

st.write('')
st.write('')
st.write('')

st.write('Your data:')

st.write(df)

# st.write(x_test_copy)
# st.write(heart_pred)

if heart_pred[0] == 1:
    st.subheader("Chances of heart disease detected")
    st.subheader('[The model detected that you might have a heart disease. Kindly refer to this link to find out what you should do next](https://www.medicalnewstoday.com/articles/257484#risk-factors)')

else:
    st.subheader("Chances of heart disease absent")

st.write('')
st.write('')
st.write('')
st.write('')

st.write('Want to learn about the model used? Click here')
model_button = st.checkbox('Know more about the model')

if model_button:
    st.subheader('This model is an artificial neural network with 3 dense layers and 20 neurons in each layer, with 91 precent training accuracy')
    st.write('')
    st.write('Heat map associated with the testing of the model:')
    test_image = Image.open(r'C:\Users\chakr\OneDrive\Desktop\output.png')
    st.image(test_image)