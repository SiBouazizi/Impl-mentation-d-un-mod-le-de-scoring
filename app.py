#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import streamlit as st
import pickle


# In[2]:


model= pickle.load(open("model_pkl", "rb"))


# In[9]:


# importer la data :
data= app_train= pd.read_csv(r"https://raw.githubusercontent.com/SiBouazizi/Impl-mentation-d-un-mod-le-de-scoring/main/train.csv", sep="\t")


# In[3]:


def predict_score(EXT_SOURCE_3, EXT_SOURCE_2, EXT_SOURCE_1, AMT_CREDIT,AMT_INCOME_TOTAL,REGION_POPULATION_RELATIVE, AMT_GOODS_PRICE,AMT_ANNUITY,DAYS_EMPLOYED,SK_ID_CURR,DAYS_REGISTRATION,DAYS_LAST_PHONE_CHANGE, HOUR_APPR_PROCESS_START, DAYS_ID_PUBLISH,DAYS_BIRTH):
    input=np.array([[EXT_SOURCE_3, EXT_SOURCE_2, EXT_SOURCE_1, AMT_CREDIT,AMT_INCOME_TOTAL, REGION_POPULATION_RELATIVE, AMT_GOODS_PRICE,AMT_ANNUITY, DAYS_EMPLOYED, SK_ID_CURR, DAYS_REGISTRATION,DAYS_LAST_PHONE_CHANGE, HOUR_APPR_PROCESS_START, DAYS_ID_PUBLISH,DAYS_BIRTH]]).astype(np.float64)  
    prediction= model.predict_proba(input)
    pred= '{0:.{1}f}'.format(prediction[0][0], 2)
    print(type(pred))
    return float(pred)


# In[4]:


def main():
    st.title("Déploiment du modèle sous forme d'une API")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Prédiction de la probabilité de défaut de paiement d'un client </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    EXT_SOURCE_3 = st.text_input("EXT_SOURCE_3","TYPE Here")
    EXT_SOURCE_2 = st.text_input("EXT_SOURCE_2","TYPE Here")
    EXT_SOURCE_1= st.text_input("EXT_SOURCE_1","TYPE Here")
    AMT_CREDIT= st.text_input("AMT_CREDIT", "TYPE Here")
    AMT_INCOME_TOTAL = st.text_input("AMT_INCOME_TOTAL","TYPE Here")
    REGION_POPULATION_RELATIVE = st.text_input("REGION_POPULATION_RELATIVE","TYPE Here")
    AMT_GOODS_PRICE = st.text_input("AMT_GOODS_PRICE","TYPE Here")
    AMT_ANNUITY = st.text_input("AMT_ANNUITY","TYPE Here")
    DAYS_EMPLOYED = st.text_input("DAYS_EMPLOYED","TYPE Here")
    SK_ID_CURR = st.text_input("SK_ID_CURR","TYPE Here")
    DAYS_REGISTRATION = st.text_input("DAYS_REGISTRATION","TYPE Here")
    DAYS_LAST_PHONE_CHANGE = st.text_input("DAYS_LAST_PHONE_CHANGE","TYPE Here")
    HOUR_APPR_PROCESS_START = st.text_input("HOUR_APPR_PROCESS_START","TYPE Here")
    DAYS_ID_PUBLISH = st.text_input("DAYS_ID_PUBLISH","TYPE Here")
    DAYS_BIRTH = st.text_input("DAYS_BIRTH","TYPE Here")

    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> L'attribution de ce crédit est sûre </h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> L'attribution de ce crédit est risqué </h2>
       </div>
    """

    if st.button("Predict"):
        output=predict_score(EXT_SOURCE_3, EXT_SOURCE_2, EXT_SOURCE_1, AMT_CREDIT,AMT_INCOME_TOTAL,REGION_POPULATION_RELATIVE, AMT_GOODS_PRICE,AMT_ANNUITY,DAYS_EMPLOYED,SK_ID_CURR,DAYS_REGISTRATION,DAYS_LAST_PHONE_CHANGE, HOUR_APPR_PROCESS_START, DAYS_ID_PUBLISH,DAYS_BIRTH)
        st.success('La probabilité que le client fait défaut dans le paiement de son crédit est {}'.format(output))

        if output > 0.5:
            st.markdown(danger_html,unsafe_allow_html=True)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()
    

