import streamlit as st
import pandas as pd
import keras
import pickle

model = keras.models.load_model('./Kerasmodel.h5')
with open('./onehot.save','rb') as f:
    onehot = pickle.load(f)
id = pd.read_csv('data_test.csv')
id = id.iloc[:,1:-1]

st.set_page_config(page_title='Deep Learning-Based Hidden Hunger Risk Prediction Platform',page_icon='./logo.png',layout='wide')

st.markdown('## <center>Deep Learning-Based Hidden Hunger Risk Prediction Platform</center>',unsafe_allow_html=True)
st.text('')
st.text('')

col1,col2,col3,col4 = st.columns(4)
with col1:
    get1 = st.selectbox('**Please select your gender**',options=['Male','Female'])
    st.text('')
    get5 = st.selectbox("**Please select your father's highest education**",options=['Less than primary school', 'Primary or junior high school', 'Senior high school or vocational school', "College or bachelor's degree", "Master's degree and above"])
    st.text('')
    get9 = st.selectbox('**Understanding of hidden hunger (HH)**',options=['Very knowledgeable', 'Somewhat knowledgeable', 'Slightly knowledgeable', 'Not knowledgeable'])
    st.text('')
    get13 = st.selectbox('**Reasons for irregular eating**',options=['Time issues', 'Habit issues', 'Emotional issues', 'Dieting for weight loss'])
    st.text('')
    get17 = st.selectbox('**Preference for strongly flavored foods**',options=['Yes','No'])
    st.text('')
    get21 = st.selectbox('**Frequency of eating fried foods**',options=['Always','Often','Seldom','Never'])
with col2:
    get2 = st.selectbox('**Please select your grade**',options=['First year of senior high school','Second year of senior high school','Third year of senior high school'])
    st.text('')
    get6 = st.selectbox("**Please select your mother's highest education**",options=['Less than primary school', 'Primary or junior high school', 'Senior high school or vocational school', "College or bachelor's degree", "Master's degree and above"])
    st.text('')
    get10 = st.selectbox('**Understanding of essential nutrients**',options=['Completely clear', 'Understand most of it', 'Know a little', "Don't understand"])
    st.text('')
    get14 = st.selectbox('**Satisfaction level with daily diet**',options=['Very satisfied', 'Satisfied', 'Somewhat dissatisfied', 'Dissatisfied'])
    st.text('')
    get18 = st.selectbox('**Taken nutritional supplements in the past year**',options=['Yes','No'])
    st.text('')
    get22 = st.selectbox('**Frequency of eating coarse grains**',options=['Always','Often','Seldom','Never'])
with col3:
    get3 = st.selectbox('**Please select your born place**',options=['Urban','Rural'])
    st.text('')
    get7 = st.selectbox('**Please select your monthly expenditure on foods**',options=['Less than 500 yuan','500-1000 yuan','1000-1500 yuan','More than 1500 yuan'])
    st.text('')
    get11 = st.selectbox('**Understanding of HH and chronic disease link**',['Completely clear', 'Understand most of it', 'Know a little', "Don't understand"])
    st.text('')
    get15 = st.selectbox('**Paying attention to nutritional balance**',options=['Yes','No'])
    st.text('')
    get19 = st.selectbox('**Please select your weekly exercise time**',options=['Less than 3 hours','3-5 hours','5-10 hours','More than 10 hours'])
    st.text('')
    get23 = st.selectbox('**Frequency of eating fruits**',options=['Always','Often','Seldom','Never'])
with col4:
    get4 = st.selectbox('**Are you an only child**',options=['Yes','No'])
    st.text('')
    get8 = st.selectbox('**Please select your daily sun exposure time**',options=['Less than 30 minutes','30-60 minutes','60-90 minutes','More than 90 minutes'])
    st.text('')
    get12 = st.selectbox('**Frequency of irregular eating**',options=['Always', 'Often', 'Seldom', 'Never'])
    st.text('')
    get16 = st.selectbox('**Is there picky eating habit**',options=['Yes','No'])
    st.text('')
    get20 = st.selectbox('**Replenishing micronutrients after sweating**',options=['Yes','No'])
    st.text('')
    get24 = st.selectbox('**Frequency of eating snacks**',options=['Always','Often','Seldom','Never'])

col5,col6,col7 = st.columns([2,2,4])
with col5:
    st.text('')
    get25 = st.selectbox('**Willingness to learn nutritional knowledge**',options=['Willing to actively learn', 'Unwilling to learn'])
with col6:
    st.text('')
    get26 = st.selectbox('**Altitude towards hidden hunger**',options=['Care',"Don't care"])
with col7:
    st.markdown('<font size=4>　</font>',unsafe_allow_html=True)
    bt = st.button('Click to predict',use_container_width=True)

if get1 == 'Male':
    get1 = 1
elif get1 == 'Female':
    get1 = 2

if get2 == 'First year of senior high school':
    get2 = 1
elif get2 == 'Second year of senior high school':
    get2 = 2
elif get2 == 'Third year of senior high school':
    get2 = 3

if get3 == 'Urban':
    get3 = 1
elif get3 == 'Rural':
    get3 = 2

if get4 == 'Yes':
    get4 = 1
elif get4 == 'No':
    get4 = 2

if get5 == 'Less than primary school':
    get5 = 1
elif get5 == 'Primary or junior high school':
    get5 = 2
elif get5 == 'Senior high school or vocational school':
    get5 = 3
elif get5 == "College or bachelor's degree":
    get5 = 4
elif get5 == "Master's degree and above":
    get5 = 5

if get6 == 'Less than primary school':
    get6 = 1
elif get6 == 'Primary or junior high school':
    get6 = 2
elif get6 == 'Senior high school or vocational school':
    get6 = 3
elif get6 == "College or bachelor's degree":
    get6 = 4
elif get6 == "Master's degree and above":
    get6 = 5

if get7 == 'Less than 500 yuan':
    get7 = 1
elif get7 == '500-1000 yuan':
    get7 = 2
elif get7 == '1000-1500 yuan':
    get7 = 3
elif get7 == 'More than 1500 yuan':
    get7 = 4

if get8 == 'Less than 30 minutes':
    get8 = 1
elif get8 == '30-60 minutes':
    get8 = 2
elif get8 == '60-90 minutes':
    get8 = 3
elif get8 == 'More than 90 minutes':
    get8 = 4

if get9 == 'Very knowledgeable':
    get9 = 1
elif get9 == 'Somewhat knowledgeable':
    get9 = 2
elif get9 == 'Slightly knowledgeable':
    get9 = 3
elif get9 == 'Not knowledgeable':
    get9 = 4

if get10 == 'Completely clear':
    get10 = 1
elif get10 == 'Understand most of it':
    get10 = 2
elif get10 == 'Know a little':
    get10 = 3
elif get10 == "Don't understand":
    get10 = 4

if get11 == 'Completely clear':
    get11 = 1
elif get11 == 'Understand most of it':
    get11 = 2
elif get11 == 'Know a little':
    get11 = 3
elif get11 == "Don't understand":
    get11 = 4

if get12 == 'Always':
    get12 = 1
elif get12 == 'Often':
    get12 = 2
elif get12 == 'Seldom':
    get12 = 3
elif get12 == "Never":
    get12 = 4

if get13 == 'Time issues':
    get13 = 1
elif get13 == 'Habit issues':
    get13 = 2
elif get13 == 'Emotional issues':
    get13 = 3
elif get13 == "Dieting for weight loss":
    get13 = 4

if get14 == 'Very satisfied':
    get14 = 1
elif get14 == 'Satisfied':
    get14 = 2
elif get14 == 'Somewhat dissatisfied':
    get14 = 3
elif get14 == "Dissatisfied":
    get14 = 4

if get15 == 'Yes':
    get15 = 1
elif get15 == 'No':
    get15 = 2

if get16 == 'Yes':
    get16 = 1
elif get16 == 'No':
    get16 = 2

if get17 == 'Yes':
    get17 = 1
elif get17 == 'No':
    get17 = 2

if get18 == 'Yes':
    get18 = 1
elif get18 == 'No':
    get18 = 2

if get19 == 'Less than 3 hours':
    get19 = 1
elif get19 == '3-5 hours':
    get19 = 2
elif get19 == '5-10 hours':
    get19 = 3
elif get19 == "More than 10 hours":
    get19 = 4

if get20 == 'Yes':
    get20 = 1
elif get20 == 'No':
    get20 = 2

if get21 == 'Always':
    get21 = 1
elif get21 == 'Often':
    get21 = 2
elif get21 == 'Seldom':
    get21 = 3
elif get21 == "Never":
    get21 = 4

if get22 == 'Always':
    get22 = 1
elif get22 == 'Often':
    get22 = 2
elif get22 == 'Seldom':
    get22 = 3
elif get22 == "Never":
    get22 = 4

if get23 == 'Always':
    get23 = 1
elif get23 == 'Often':
    get23 = 2
elif get23 == 'Seldom':
    get23 = 3
elif get23 == "Never":
    get23 = 4

if get24 == 'Always':
    get24 = 1
elif get24 == 'Often':
    get24 = 2
elif get24 == 'Seldom':
    get24 = 3
elif get24 == "Never":
    get24 = 4

if get25 == 'Willing to actively learn':
    get25 = 1
elif get25 == 'Unwilling to learn':
    get25 = 2

if get26 == 'Care':
    get26 = 1
elif get26 == "Don't care":
    get26 = 2

if bt:
    df = pd.DataFrame({
        'gender':[get1],
        'grade':[get2],
        'origin':[get3],
        'onlychild':[get4],
        'faedu':[get5],
        'maedu':[get6],
        'cost':[get7],
        'time':[get8],
        'q1':[get9],
        'q2':[get10],
        'q3':[get11],
        'q4':[get12],
        'q5':[get13],
        'q6':[get14],
        'q7':[get15],
        'q8':[get16],
        'q9':[get17],
        'q10':[get18],
        'q11':[get19],
        'q12':[get20],
        'q13':[get21],
        'q14':[get22],
        'q15':[get23],
        'q16':[get24],
        'q17':[get25],
        'q18':[get26]
    })
    df_onehot = pd.DataFrame(onehot.transform(df).toarray())
    df_onehot.columns = id.columns
    pre = model.predict(df_onehot)
    st.text('')
    st.text('')
    if pre >= 0.45:
        st.markdown('## <center>Your Risk of Hidden Hunger is High</center>',unsafe_allow_html=True)
    elif pre < 0.45:
        st.markdown('## <center>Your Risk of Hidden Hunger is Low</center>',unsafe_allow_html=True)