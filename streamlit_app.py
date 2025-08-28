import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import sys
import os

# ---------------------------
# App title & helper text
# ---------------------------
st.set_page_config(page_title="🏠 House Price Prediction", layout="wide")
st.title("🏠 House Price Prediction App")
st.caption("Upload an Excel file with the required columns to generate predictions from multiple trained models.")

# ---------------------------
# Environment/version display (helps debug joblib/pickle mismatches)
# ---------------------------
with st.expander("ℹ️ Environment info (helpful for debugging)"):
    try:
        import sklearn, xgboost
        st.write({
            "Python": sys.version.split()[0],
            "joblib": joblib.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "NumPy": np.__version__,
            "Pandas": pd.__version__,
        })
    except Exception as e:
        st.write("Could not import all optional libs:", e)

# ---------------------------
# Load training data (for ranges display only)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_train_data():
    url = "https://raw.githubusercontent.com/rr2734/rashmir/refs/heads/main/train.csv"
    return pd.read_csv(url)

train_data = load_train_data()

numeric_cols_display = [
    'Fireplaces', 'GarageYrBlt','WoodDeckSF', 'OpenPorchSF', '2ndFlrSF','MasVnrArea',
    'BsmtFinSF1', 'LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd',
    'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd',
    'GarageCars','GarageArea', 'SalePrice'
]
numeric_cols = [c for c in numeric_cols_display if c != 'SalePrice']  # used for validation

with st.expander("📋 View required numeric column ranges"):
    if train_data is not None:
        numeric_ranges = {col: (train_data[col].min(), train_data[col].max())
                          for col in numeric_cols if col in train_data.columns}
        for col, (mn, mx) in numeric_ranges.items():
            st.write(f"- **{col}**: {mn} to {mx}")
    else:
        st.info("Training data not available to compute ranges.")

# ---------------------------
# Load models & artifacts
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_artifacts():
    """Load all serialized models and the scaler. Raises a helpful error if something mismatches."""
    model_files = {
        "model_sk": "multilinear_regression_model.pkl",
        "tree_model": "decision_tree_regressor.pkl",
        "scaler": "scaler.pkl",
        "adaboost_regressor": "adaboost_regressor.pkl",
        "bagging_model": "bagging_model.pkl",
        "gnb": "gnb.pkl",
        "gradientboostingmodel": "gradientboostingmodel.pkl",
        "randomforestmodel": "randomforestmodel.pkl",
        "xgb_model": "xgb_model.pkl",
    }

    loaded = {}
    for key, fname in model_files.items():
        if not os.path.exists(fname):
            raise FileNotFoundError(
                f"Missing file: {fname}. "
                "Make sure it is in the same folder as this app on Streamlit Cloud (Manage app → Files)."
            )
        try:
            with open(fname, "rb") as f:
                loaded[key] = joblib.load(f)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load '{fname}' with joblib. "
                "This often means Python / scikit-learn / xgboost versions differ between training and deployment. "
                "Match versions or export XGBoost with model.save_model('xgb.json')."
            ) from e

    # train_features.json must be a JSON list of column names used at train time
    tf_path = "train_features.json"
    if not os.path.exists(tf_path):
        raise FileNotFoundError(
            "Missing file: train_features.json (list of feature columns used during training)."
        )
    with open(tf_path, "r") as f:
        train_features = json.load(f)
    if not isinstance(train_features, list):
        raise ValueError("train_features.json must contain a JSON list of feature names.")

    return loaded, train_features

try:
    artifacts, train_features = load_artifacts()
    model_sk = artifacts["model_sk"]
    tree_model = artifacts["tree_model"]
    scaler = artifacts["scaler"]
    adaboost_regressor = artifacts["adaboost_regressor"]
    bagging_model = artifacts["bagging_model"]
    gnb = artifacts["gnb"]
    gradientboostingmodel = artifacts["gradientboostingmodel"]
    randomforestmodel = artifacts["randomforestmodel"]
    xgb_model = artifacts["xgb_model"]
except Exception as e:
    st.error(f"❌ Problem loading models/artifacts: {e}")
    st.stop()

# ---------------------------
# File upload
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload Excel file", type=["xlsx"])

# ---------------------------
# Prediction pipeline
# ---------------------------
def prepare_features(df: pd.DataFrame, feature_list: list, scaler):
    """
    Validate, select numeric columns, align to training feature order, and apply the pre-fitted scaler.
    Returns (test_final, id_col).
    """
    # Ensure 'Id' exists (if not, create a simple index-based Id)
    if 'Id' not in df.columns:
        df = df.copy()
        df['Id'] = np.arange(1, len(df) + 1)

    # Numeric cols in uploaded df (exclude target-like columns)
    numeric_in_df = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['SalePrice']]

    # Create numerical matrix from available numeric columns
    numerical_features = df[numeric_in_df].copy()

    # Ensure all train features are present (add missing with zeros)
    for col in feature_list:
        if col not in numerical_features.columns:
            numerical_features[col] = 0

    # Keep only features used at train time, in the exact order
    numerical_features = numerical_features[feature_list]

    # Apply pre-fitted scaler (DO NOT fit on test!)
    X = pd.DataFrame(
        scaler.transform(numerical_features),
        columns=feature_list,
        index=numerical_features.index
    )

    id_col = df.loc[X.index, 'Id']
    # Drop rows with NaN if any slipped in
    notna_idx = X.dropna(axis=0).index
    X = X.loc[notna_idx]
    id_col = id_col.loc[notna_idx]
    return X, id_col

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)

        # Basic validation for numeric columns shown in the UI
        missing_numeric = [col for col in numeric_cols if col not in df.columns]
        if missing_numeric:
            st.error(f"Missing required numeric columns (as per training set display): {missing_numeric}")
            st.stop()
        else:
            st.success("✅ File loaded successfully!")

        # Build features aligned with training
        test_final, id_col = prepare_features(df, train_features, scaler)

        # ---------------------------
        # Predictions (keep model objects intact)
        # ---------------------------
        preds = {}

        # Linear Regression (appears to be trained on log-target)
        linear_preds = model_sk.predict(test_final)
        linear_preds = np.exp(linear_preds) - 1
        linear_preds = np.maximum(linear_preds, 0)
        preds['Linear_Regression_Pred'] = linear_preds

        # Decision Tree
        preds['Decision_Tree_Pred'] = tree_model.predict(test_final)

        # Random Forest
        preds['Random_Forest_Regressor_Pred'] = randomforestmodel.predict(test_final)

        # XGBoost
        preds['XGBoost_Regressor_Pred'] = xgb_model.predict(test_final)

        # Gaussian Naive Bayes (note: classifier outputs labels; ensure your training made sense)
        try:
            preds['Gaussian_Naive_Bayes_Model_Pred'] = gnb.predict(test_final)
        except Exception as e:
            preds['Gaussian_Naive_Bayes_Model_Pred'] = np.full(len(test_final), np.nan)

        # Bagging
        preds['Bagging_Predictions'] = bagging_model.predict(test_final)

        # AdaBoost
        preds['Adaboost_Regressor_Predictions'] = adaboost_regressor.predict(test_final)

        # Gradient Boosting
        preds['Gradient_Boosting_Predictions'] = gradientboostingmodel.predict(test_final)

        # ---------------------------
        # Assemble results
        # ---------------------------
        results_df = pd.DataFrame({"Id": id_col.values})
        for col, arr in preds.items():
            results_df[col] = arr

        st.subheader("🔮 Predictions")
        st.dataframe(results_df, use_container_width=True)

        # Download as Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            results_df.to_excel(writer, index=False, sheet_name="predictions")
        st.download_button(
            label="⬇️ Download Predictions as Excel",
            data=output.getvalue(),
            file_name="predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.stop()
else:
    st.info("Upload an Excel file to begin.")
