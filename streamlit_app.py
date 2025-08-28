import streamlit as st
import pickle
import io
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor 

# --- Load your saved models ---
with open('multilinear_regression_model.pkl', 'rb') as f:
    model_sk = joblib.load(f)

with open('decision_tree_regressor.pkl', 'rb') as f:
    tree_model = joblib.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)
with open('adaboost_regressor.pkl', 'rb') as f:
    adaboost_regressor = joblib.load(f)
with open('bagging_model.pkl', 'rb') as f:
    bagging_model = joblib.load(f)
with open('gnb.pkl', 'rb') as f:
    gnb = joblib.load(f)
with open('gradientboostingmodel.pkl', 'rb') as f:
    gradientboostingmodel = joblib.load(f)
with open('randomforestmodel.pkl', 'rb') as f:
    randomforestmodel = joblib.load(f)
with open('xgb_model.pkl', 'rb') as f:
    xgb_model = joblib.load(f)


train_data = pd.read_csv('https://raw.githubusercontent.com/rr2734/rashmir/refs/heads/main/train.csv')  # path to your original training file

# Identify numeric and categorical columns
numeric_cols1 = ['Fireplaces', 'GarageYrBlt','WoodDeckSF', 
                                        'OpenPorchSF', '2ndFlrSF','MasVnrArea',
                                        'BsmtFinSF1', 'LotFrontage', 'OverallQual', 
                                        'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 
                                        '1stFlrSF', 'GrLivArea', 'FullBath', 
                                        'TotRmsAbvGrd', 'GarageCars','GarageArea', 'SalePrice']

numeric_cols1.remove('SalePrice')

# Compute ranges for numeric columns
numeric_ranges = {col: (train_data[col].min(), train_data[col].max()) for col in numeric_cols1}



# ----------------------
# 3. Display ranges and options
# ----------------------
st.title("🏠 House Price Prediction App")
st.subheader("Upload your test data Excel file")

with st.expander("📋 View required column ranges and categories"):
    st.write("**Numeric column ranges:**")
    for col, (min_val, max_val) in numeric_ranges.items():
        st.write(f"- {col}: {min_val} to {max_val}")

 
result = pd.read_csv('https://raw.githubusercontent.com/rr2734/rashmir/refs/heads/main/train.csv')
numeric_cols=['Fireplaces', 'GarageYrBlt','WoodDeckSF', 
                                        'OpenPorchSF', '2ndFlrSF','MasVnrArea',
                                        'BsmtFinSF1', 'LotFrontage', 'OverallQual', 
                                        'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 
                                        '1stFlrSF', 'GrLivArea', 'FullBath', 
                                        'TotRmsAbvGrd', 'GarageCars','GarageArea', 'SalePrice']
numeric_cols.remove('SalePrice')





uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file)

        # --- Validate columns ---
        missing_numeric = [col for col in numeric_cols if col not in df.columns]
       

        if missing_numeric:
            st.error(f"Missing required columns: {missing_numeric}")
        else:
            st.success("File loaded successfully!")

        with open('train_features.json') as f:
            train_features = json.load(f)

        id_col = test_data['Id']
        numeric_cols = [col for col in test_data.select_dtypes(include=['int64', 'float64']).columns if col not in ['SalePrice', 'Id']]
     
       
     
        numerical_features = test_encoded[numeric_cols]
        scaled_numerical_features = pd.DataFrame(scaler.fit_transform(numerical_features), 
                                         columns=numerical_features.columns, 
                                         index=numerical_features.index)
        test_final = scaled_numerical_features
        test_final=test_final.dropna(axis=0)
        id_col = id_col.loc[test_final.index]

        for col in train_features:
            if col not in test_final.columns:
                test_final[col] = 0

        test_final= test_final[train_features]
       



            
        # --- Make predictions ---
        linear_preds = model_sk.predict(test_final)
        linear_preds = np.exp(linear_preds) - 1
        linear_preds = np.maximum(linear_preds, 0)

        #tree model
        tree_preds = tree_model.predict(test_final)

        #random forest predictions
        randomforestpred = randomforestmodel.predict(test_final)
        #Xgboost regressor
        xgbpredictions = xgb_model.predict(test_final)
        #Gaussian Naive Bayes Model
        gnbpredict = gnb.predict(test_final)
        #Bagging
        baggingpred = bagging_model.predict(test_final)
        #Adaboost Regressor
        adaboost = adaboost_regressor.predict(test_final)
        gradientboostingmodel = gradientboostingmodel.predict(test_final)
        
        # --- Show results ---
        results_df = test_final.copy()
        results_df['Id'] = id_col
        results_df['Linear_Regression_Pred'] = linear_preds
        results_df['Decision_Tree_Pred'] = tree_preds
        results_df['Random_Forest_Regressor_Pred'] = randomforestpred
        results_df['XGBoost_Regressor_Pred'] = xgbpredictions
        results_df['Gaussian_Naive_Bayes_Model_Pred']=gnbpredict
        results_df['Bagging_Predictions']=baggingpred
        results_df['Adaboost_Regressor_Predictions'] = adaboost
        result_df['Gradient_Boosting_Predictions']=gradientboostingmodel

        st.subheader("Predictions")
        st.dataframe(results_df)

        # --- Download option ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            results_df.to_excel(writer, index=False)
        st.download_button(
        label="Download Predictions as Excel",
        data=output.getvalue(),
        file_name="predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except Exception as e:

        st.error(f"Error reading file: {e}")



































