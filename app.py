import streamlit as st
from Applied_ML import Classifier, Regressor
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------
st.set_page_config(page_title='Applied ML App',
    layout='wide')

st.write("""
# Applied ML App
""")


add_sidebar = st.sidebar.selectbox('Select type of problem', ('REGRESSION', 'CLASSIFICATION'))

data = st.file_uploader('Upload a CSV')
target_column_name = st.text_input('Please type Target column name here')


if add_sidebar == 'REGRESSION':
    if st.button("Build"):
        if not data:
            st.write("Please upload a csv file")
        else:
            if not target_column_name:
                st.write("Please write a target column name")
            else:
                df = pd.read_csv(data)
                X = df.drop(target_column_name, axis=1)
                y = df[target_column_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.3,random_state = 143)
                reg = Regressor(verbose=0, ignore_warnings=False, custom_metric=None)
                models, predictions = reg.fit(X_train, X_test, y_train, y_test)
                st.table(models)

if add_sidebar == 'CLASSIFICATION':
    if st.button("Build"):
        if not data:
            st.write("Please upload a csv file")
        else:
            if not target_column_name:
                st.write("Please write a target column name")
            else:
                df = pd.read_csv(data)
                X = df.drop(target_column_name, axis=1)
                y = df[target_column_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.3,random_state = 143)
                reg = Classifier(verbose=0, ignore_warnings=False, custom_metric=None)
                models, predictions = reg.fit(X_train, X_test, y_train, y_test)
                st.table(models)