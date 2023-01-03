import socket
import os
import shutil
import pandas as pd
import numpy as np
from glob import glob
from threading import Timer
import time
import pickle
import streamlit as st
from scipy.stats import skew

st.set_page_config(layout="wide")

st.title("Smart HCI_Eye Movement_AI model implementation")
st.write(socket.gethostname())

hide_table_row_index = """
        <style>
        thead tr th:first-child {display:none}
        tbody th {display:none}
        </style>
        """
st.markdown(hide_table_row_index, unsafe_allow_html=True)

FPOGX_array = []
FPOGY_array = []
FPOGD_array = []
LPCX_array = []
LPCY_array = []
LPD_array = []
RPCX_array = []
RPCY_array = []
RPD_array = []
BKDUR_array = []
BKPMIN_array = []
LPMM_array = []
RPMM_array = []

r = st.empty()
t = st.empty()
x = st.empty()
y = st.empty()
z = st.empty()
a = []
b = 1

# Host machine IP
HOST = 'srv-celrsjen6mpkfa5tmgn0-kn-00014-deployment-9b568bfdf-5qw4q'
# Gazepoint Port
PORT = 4242
ADDRESS = (HOST, PORT)

# Connect to Gazepoint API
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(ADDRESS)

# Send commands to initialize data streaming
s.send(str.encode('<SET ID="ENABLE_SEND_CURSOR" STATE="1" />\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_POG_FIX" STATE="1" />\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_PUPIL_LEFT" STATE="1" />\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_PUPIL_RIGHT" STATE="1" />\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_BLINK" STATE="1" />\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_PUPILMM" STATE="1" />\r\n'))

@st.cache(allow_output_mutation=True, show_spinner=False)
def predict(model, a):
    prediction = model.predict([a])[0]
    if prediction == 0:
        result = 'Reading'
    elif prediction == 1:
        result = 'Watching'
    else:
        result = 'Typing'
    return result

model1 = pickle.load(open('randomforest_model.pkl', 'rb'))
model2 = pickle.load(open('decisiontree_model.pkl', 'rb'))
model3 = pickle.load(open('svm_model.pkl', 'rb'))

while 1:
    # Receive data
    rxdat = s.recv(1024)
    data = bytes.decode(rxdat)

    # Parse data string
    FPOGX = 0
    FPOGY = 0
    FPOGD = 0
    LPCX = 0
    LPCY = 0
    LPD = 0
    RPCX = 0
    RPCY = 0
    RPD = 0
    BKDUR = 0
    BKPMIN = 0
    LPMM = 0
    RPMM = 0
    
    # Split data string into a list of name="value" substrings
    datalist = data.split(" ")

    # Iterate through list of substrings to extract data values
    for el in datalist:
        try:
            if (el.find("FPOGX") != -1):
                FPOGX = float(el.split("\"")[1])
                FPOGX_array.append(FPOGX)
        except:
            FPOGX_array.append(0)
        try:
            if (el.find("FPOGY") != -1):
                FPOGY = float(el.split("\"")[1])
                FPOGY_array.append(FPOGY)
        except:
            FPOGX_array.append(0)
        try:
            if (el.find("FPOGD") != -1):
                FPOGD = float(el.split("\"")[1])
                FPOGD_array.append(FPOGD)
        except:
            FPOGX_array.append(0)
        try:
            if (el.find("LPCX") != -1):
                LPCX = float(el.split("\"")[1])
                LPCX_array.append(LPCX)
        except:
            FPOGX_array.append(0)
        try:    
            if (el.find("LPCY") != -1):
                LPCY = float(el.split("\"")[1])
                LPCY_array.append(LPCY)
        except:
            FPOGX_array.append(0)
        try:    
            if (el.find("LPD") != -1):
                LPD = float(el.split("\"")[1])
                LPD_array.append(LPD)
        except:
            FPOGX_array.append(0)
        try:    
            if (el.find("RPCX") != -1):
                RPCX = float(el.split("\"")[1])
                RPCX_array.append(RPCX)
        except:
            FPOGX_array.append(0)
        try:    
            if (el.find("RPCY") != -1):
                RPCY = float(el.split("\"")[1])
                RPCY_array.append(RPCY)
        except:
            FPOGX_array.append(0)
        try:    
            if (el.find("RPD") != -1):
                RPD = float(el.split("\"")[1])
                RPD_array.append(RPD)
        except:
            FPOGX_array.append(0)
        try:    
            if (el.find("BKDUR") != -1):
                BKDUR = float(el.split("\"")[1])
                BKDUR_array.append(BKDUR)
        except:
            FPOGX_array.append(0)
        try:    
            if (el.find("BKPMIN") != -1):
                BKPMIN = float(el.split("\"")[1])
                BKPMIN_array.append(BKPMIN)
        except:
            FPOGX_array.append(0)
        try:    
            if (el.find("LPMM") != -1):
                LPMM = float(el.split("\"")[1])
                LPMM_array.append(LPMM)
        except:
            FPOGX_array.append(0)
        try:    
            if (el.find("RPMM") != -1):
                RPMM = float(el.split("\"")[1])
                RPMM_array.append(RPMM)
        except:
            FPOGX_array.append(0)
        if len(FPOGX_array) == 150:
            df = pd.DataFrame({
                'FPOGX':[f'{round(np.array(FPOGX_array).mean(), 4)}'],
                'FPOGY':[f'{round(np.array(FPOGY_array).mean(), 4)}'],
                'FPOGD':[f'{round(np.array(FPOGD_array).mean(), 4)}'],
                'LPCX':[f'{round(np.array(LPCX_array).mean(), 4)}'],
                'LPCY':[f'{round(np.array(LPCY_array).mean(), 4)}'],
                'LPD':[f'{round(np.array(LPD_array).mean(), 4)}'],
                'RPCX':[f'{round(np.array(RPCX_array).mean(), 4)}'],
                'RPCY':[f'{round(np.array(RPCY_array).mean(), 4)}'],
                'RPD':[f'{round(np.array(RPD_array).mean(), 4)}'],
                'BKDUR':[f'{round(np.array(BKDUR_array).mean(), 4)}'],
                'BKPMIN':[f'{round(np.array(BKPMIN_array).mean(), 4)}'],
                'LPMM':[f'{round(np.array(LPMM_array).mean(), 4)}'],
                'RPMM':[f'{round(np.array(RPMM_array).mean(), 4)}'],
                'SACCADE_MAG':[0],
                'SACCADE_DIR':[0],
            })

            a = []
            a.append(np.array(FPOGD_array).mean())
            a.append(np.array(LPD_array).mean())
            a.append(np.array(RPD_array).mean())
            a.append(np.array(BKDUR_array).mean())
            a.append(np.array(BKPMIN_array).mean())
            a.append(np.array(LPMM_array).mean())
            a.append(np.array(RPMM_array).mean())
            a.append(0)
            a.append(np.array(FPOGD_array).std())
            a.append(np.array(LPD_array).std())
            a.append(np.array(RPD_array).std())
            a.append(np.array(BKDUR_array).std())
            a.append(np.array(BKPMIN_array).std())
            a.append(np.array(LPMM_array).std())
            a.append(np.array(RPMM_array).std())
            a.append(0)
            a.append(np.array(FPOGD_array).var())
            a.append(np.array(LPD_array).var())
            a.append(np.array(RPD_array).var())
            a.append(np.array(BKDUR_array).var())
            a.append(np.array(BKPMIN_array).var())
            a.append(np.array(LPMM_array).var())
            a.append(np.array(RPMM_array).var())
            a.append(0)
            a.append(np.median(np.array(FPOGD_array)))
            a.append(np.median(np.array(LPD_array)))
            a.append(np.median(np.array(RPD_array)))
            a.append(np.median(np.array(BKDUR_array)))
            a.append(np.median(np.array(BKPMIN_array)))
            a.append(np.median(np.array(LPMM_array)))
            a.append(np.median(np.array(RPMM_array)))
            a.append(0)
            a.append(np.array(LPCX_array).max())
            a.append(np.array(RPCX_array).max())
            a.append(np.array(BKDUR_array).max())
            a.append(np.array(BKPMIN_array).max())
            a.append(np.array(LPMM_array).max())
            a.append(np.array(RPMM_array).max())
            a.append(0)
            a.append(0)
            a.append(np.array(FPOGD_array).min())
            a.append(np.array(LPD_array).min())
            a.append(np.array(RPD_array).min())
            a.append(np.array(BKDUR_array).min())
            a.append(np.array(BKPMIN_array).min())
            a.append(np.array(LPMM_array).min())
            a.append(np.array(RPMM_array).min())
            a.append(0)
            a.append(skew(FPOGD_array))
            a.append(skew(LPD_array))
            a.append(skew(RPD_array))
            a.append(skew(BKDUR_array))
            a.append(skew(BKPMIN_array))
            a.append(skew(LPMM_array))
            a.append(skew(RPMM_array))
            a.append(0)

            r.table(df)
            result1 = predict(model1, a)
            result2 = predict(model2, a)
            result3 = predict(model3, a)
            x.markdown(f"<h2 style='text-align: center; color: black;'>Random Forest Result: {result1}</h2>", unsafe_allow_html=True)
            y.markdown(f"<h2 style='text-align: center; color: black;'>Decision Tree Result: {result2}</h2>", unsafe_allow_html=True)
            z.markdown(f"<h2 style='text-align: center; color: black;'>SVM Result: {result3}</h2>", unsafe_allow_html=True)
            FPOGX_array = []
            FPOGY_array = []
            FPOGD_array = []
            LPCX_array = []
            LPCY_array = []
            LPD_array = []
            RPCX_array = []
            RPCY_array = []
            RPD_array = []
            BKDUR_array = []
            BKPMIN_array = []
            LPMM_array = []
            RPMM_array = []

    b = b+1
    if t.button('Stop', key=b):
        s.close()
        break
