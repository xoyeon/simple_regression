
#streamlit 라이브러리를 불러오기
import streamlit as st
#AI모델을 불러오기 위한 joblib 불러오기
import joblib
import pandas as pd

# st를 이용하여 타이틀과 입력 방법을 명시한다.

def user_input_features() :
  signalstrenth = st.slider("신호 강도: ", 4.0, 8.0)
  antenacoverage = st.slider("안테나 커버리지: ", 2.0, 4.5)
  antenalength = st.slider("안테나 길이: ", 0.0, 7.0)
  bandwidth = st.slider("밴드범위: ", 0.0 ,2.5)

  data = {'signalstrenth' : [signalstrenth],
          'antenacoverage' : [antenacoverage],
          'antenalength' : [antenalength],
          'bandwidth' : [bandwidth],
          }

  data_df = pd.DataFrame(data, index=[0])
  return data_df

# 'signalstrenth', 'antenacoverage', 'antenalength', 'bandwidth', 'VOC'

st.title('네트워크 VOC 예측')
st.markdown('아래의 slider를 움직여 값을 입력하시오')
st.caption('민원내용에 따른 등급: 0은 aclass, 1는 bclass, 2는 sclass 민원')

le = joblib.load("le.pkl")
scaler_call = joblib.load("scaler.pkl")
model_call = joblib.load("model.pkl")


#y값을 변환할때 사용
#머신러닝으로 저장된 모델을 호출하고 st로 부터 받은 값으로 예측한다.
new_x_df = user_input_features()

new_x_transformed = scaler_call.transform(new_x_df)
new_pred = model_call.predict(new_x_transformed)

#예측결과를 화면에 뿌려준다. 
st.subheader('결과는 다음과 같습니다.')
st.write('민원 내용:', new_pred[0])
