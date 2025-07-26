import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import unicodedata
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from datetime import datetime
import pytz
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
import os

# Initial configuration
st.title("🔍 Annual GDP Prediction for 2024")
cet = pytz.timezone('CET')
current_date_time = cet.localize(datetime(2025, 7, 26, 23, 19))  # Updated to 11:19 PM CET
st.write(f"**Current Date and Time:** {current_date_time.strftime('%d/%m/%Y %H:%M %Z')}")

# Set random seed
random.seed(42)
np.random.seed(42)

# Initialize error log
error_log = []

# Normalize string function
def normalize_name(name):
    if pd.isna(name) or not isinstance(name, str):
        error_log.append(f"Non-text or NaN value: {name}. Replaced with 'unknown'.")
        return "unknown"
    original_name = name
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8').strip()
    name = re.sub(r"['’´]+", "'", name)
    name = re.sub(r'\s+', ' ', name).lower()
    name = name.replace("d'autre produits", "d'autres produits")
    name = name.replace("de lhabillement", "de l'habillement")
    name = name.replace("crise sociale", "social crisis")
    if name.startswith("impots nets de subventions") or name.startswith("impôts nets de subventions"):
        name = "net taxes on products"
        error_log.append(f"Normalized '{original_name}' to 'net taxes on products'.")
    error_log.append(f"Normalization: '{original_name}' -> '{name}'")
    return name

# Load and preprocess data
@st.cache_data
def load_and_preprocess(uploaded_file=None):
    try:
        if uploaded_file:
            # Reset file pointer and read raw content for debugging
            uploaded_file.seek(0)
            raw_content = uploaded_file.read().decode('utf-8')
            if not raw_content.strip():
                error_log.append("The uploaded file is empty.")
                st.error("The uploaded file is empty. Please check the file.")
                raise ValueError("Empty file.")
            error_log.append(f"Raw content of uploaded file: {raw_content[:200]}...")
            
            # Try reading with different separators
            for sep in [';', ',', '\t']:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=sep, encoding='utf-8')
                    if not df.empty and 'année' in df.columns:
                        error_log.append(f"File loaded with separator '{sep}'.")
                        break
                except Exception as e:
                    error_log.append(f"Failed to read with separator '{sep}': {str(e)}")
            else:
                error_log.append("Unable to read the file with tested separators (; , \\t).")
                st.error("Unable to read the CSV file. Check the format and separator.")
                raise ValueError("Invalid CSV format or incorrect separator.")
        else:
            # Fallback to default file
            default_file = "VA-2015-2023P.csv"
            if not os.path.exists(default_file):
                error_log.append(f"File '{default_file}' not found.")
                st.error(f"File '{default_file}' not found. Check the file path.")
                raise FileNotFoundError(f"File '{default_file}' not found.")
            df = pd.read_csv(default_file, sep=';', encoding='utf-8')
            error_log.append(f"File loaded as CSV with separator ';'.")

        # Validate DataFrame
        if df.empty or len(df.columns) == 0:
            error_log.append("The CSV file contains no valid columns.")
            st.error("The CSV file contains no valid columns. Check the file content.")
            raise ValueError("No columns in the CSV file.")
        if 'année' not in df.columns:
            error_log.append(f"Column 'année' missing. Found columns: {df.columns.tolist()}")
            st.error(f"The 'année' column is required. Found columns: {df.columns.tolist()}")
            raise ValueError("Column 'année' missing.")

        df = df.rename(columns={'année': 'Sector'})
        error_log.append(f"Raw sectors in CSV: {df['Sector'].tolist()}")
        df['Sector'] = df['Sector'].apply(normalize_name)
        error_log.append(f"Sectors after normalization: {df['Sector'].tolist()}")

        for col in df.columns[1:]:
            df[col] = df[col].astype(str).str.replace(' ', '').str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        sectors = [
            "agriculture, forestry and fishing",
            "oil and natural gas extraction",
            "mining products extraction",
            "food and beverage industries",
            "textile, clothing and leather industry",
            "oil refining",
            "chemical industries",
            "other non-metallic mineral products industry",
            "mechanical and electrical industries",
            "various industries",
            "electricity and gas production and distribution",
            "water production and distribution and waste management",
            "construction",
            "trade and repair",
            "transport and storage",
            "accommodation and catering",
            "information and communication",
            "financial and insurance activities",
            "public administration and defense",
            "education",
            "human health and social work",
            "other market services",
            "other household activities",
            "activities of associative organizations"
        ]
        macro_keywords = [
            "unemployment rate", "inflation rate", "interest rate", "public debt", "tax pressure",
            "international monetary policy", "regional geopolitical tensions", "commodity prices",
            "drought and climate disaster", "pandemics", "social crisis",
            "net taxes on products"
        ]
        macro_rates = ["unemployment rate", "inflation rate", "interest rate", "public debt", "tax pressure"]
        events = [
            "international monetary policy", "regional geopolitical tensions", "commodity prices",
            "drought and climate disaster", "pandemics", "social crisis"
        ]

        if 'f' in df['Sector'].values:
            error_log.append("Row 'f' detected in sectors. It will be excluded.")
            df = df[df['Sector'] != 'f']

        if not df['Sector'].str.contains("gross domestic product gdp", case=False).any():
            st.error("No GDP data found. Available sectors: {}".format(df['Sector'].tolist()))
            error_log.append("No GDP data found in the file.")
            raise ValueError("GDP data missing.")

        taxes_key = "net taxes on products"
        df_macro = df[df['Sector'].isin(macro_keywords)].copy()
        df_pib = df[df['Sector'] == "gross domestic product gdp"].copy()
        df_secteurs = df[df['Sector'].isin(sectors)].copy()
        df_secteurs = df_secteurs[df_secteurs['Sector'] != taxes_key]

        if taxes_key not in df_macro['Sector'].values:
            error_log.append(f"Error: '{taxes_key}' not found in df_macro.")
            st.error(f"Error: '{taxes_key}' not found in df_macro.")
            raise ValueError(f"'{taxes_key}' missing in df_macro.")
        if taxes_key in df_secteurs['Sector'].values:
            error_log.append(f"Error: '{taxes_key}' found in df_secteurs after exclusion.")
            st.error(f"Error: '{taxes_key}' found in df_secteurs after exclusion.")
            raise ValueError(f"'{taxes_key}' found in df_secteurs after exclusion.")

        error_log.append(f"Sectors in df_secteurs: {df_secteurs['Sector'].tolist()}")
        error_log.append(f"Macros in df_macro: {df_macro['Sector'].tolist()}")

        if df_pib.empty:
            st.error("No GDP data found. Available sectors: {}".format(df['Sector'].tolist()))
            error_log.append("No GDP data found in the file.")
            raise ValueError("GDP data missing.")

        missing_sectors = [s for s in sectors if s not in df['Sector'].values]
        missing_macro = [m for m in macro_keywords if m not in df['Sector'].values]
        if missing_sectors:
            st.warning(f"Missing sectors: {missing_sectors}. Using average of available sectors.")
            error_log.append(f"Missing sectors: {missing_sectors}")
        if missing_macro:
            st.warning(f"Missing macros: {missing_macro}. Using default values (0).")
            error_log.append(f"Missing macros: {missing_macro}")

        df_macro.set_index("Sector", inplace=True)
        df_pib.set_index("Sector", inplace=True)
        df_secteurs.set_index("Sector", inplace=True)

        df_macro_T = df_macro.transpose()
        df_secteurs_T = df_secteurs.transpose()
        df_pib_T = df_pib.transpose()

        X_df = pd.concat([df_secteurs_T, df_macro_T[macro_rates + events]], axis=1).dropna()
        y_df = df_pib_T.loc[X_df.index]

        error_log.append(f"Columns in X_df after concatenation: {list(X_df.columns)}")

        if y_df.empty:
            st.error("y_df empty after alignment with X_df. X_df indices: {}. df_pib_T indices: {}".format(X_df.index.tolist(), df_pib_T.index.tolist()))
            error_log.append("y_df empty after alignment.")
            raise ValueError("GDP data empty after preprocessing.")

        key_sectors = [
            "agriculture, forestry and fishing", "mechanical and electrical industries",
            "accommodation and catering", "information and communication",
            "financial and insurance activities"
        ]
        for sector in key_sectors:
            if sector in X_df.columns:
                X_df[f"{sector}_lag1"] = X_df[sector].shift(1).fillna(X_df[sector].mean())
            else:
                X_df[f"{sector}_lag1"] = X_df[sectors].mean(axis=1).shift(1).fillna(X_df[sectors].mean().mean()) if sectors else 0
                error_log.append(f"Lagged feature '{sector}_lag1' added with average of sectors since '{sector}' is missing.")

        for rate in macro_rates:
            if rate in X_df.columns:
                X_df[f"{rate}_lag1"] = X_df[rate].shift(1).fillna(X_df[rate].mean())
            else:
                X_df[f"{rate}_lag1"] = 0
                error_log.append(f"Lagged feature '{rate}_lag1' added with value 0 since '{rate}' is missing.")

        X_df['gdp_lag1'] = y_df.shift(1).fillna(y_df.mean())

        expected_features = sectors + macro_rates + events + [f"{s}_lag1" for s in key_sectors] + [f"{r}_lag1" for r in macro_rates] + ['gdp_lag1']
        error_log.append(f"Expected columns in X_df: {expected_features} (count: {len(expected_features)})")

        missing_cols = [col for col in expected_features if col not in X_df.columns]
        extra_cols = [col for col in X_df.columns if col not in expected_features]
        if missing_cols:
            existing_cols = [col for col in sectors + macro_rates + events if col in X_df.columns]
            for col in missing_cols:
                if col in sectors and existing_cols:
                    X_df[col] = X_df[existing_cols].mean(axis=1)
                    error_log.append(f"Missing feature '{col}' added with average of available sectors.")
                elif col.endswith('_lag1') and col.replace('_lag1', '') in X_df.columns:
                    X_df[col] = X_df[col.replace('_lag1', '')].shift(1).fillna(X_df[col.replace('_lag1', '')].mean())
                    error_log.append(f"Missing feature '{col}' added with lag.")
                else:
                    X_df[col] = 0
                    error_log.append(f"Missing feature '{col}' added with value 0.")
        if extra_cols:
            st.warning(f"Extra columns in X_df: {extra_cols}")
            error_log.append(f"Extra columns in X_df: {extra_cols}")
            X_df = X_df.drop(columns=extra_cols, errors='ignore')

        error_log.append(f"Columns in X_df after adding missing features: {list(X_df.columns)}")
        error_log.append(f"Number of columns in X_df: {X_df.shape[1]} (expected: {len(expected_features)})")

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_df = X_df[expected_features]
        scaler_X.fit(X_df)
        error_log.append(f"Scaler_X fitted on {scaler_X.n_features_in_} features")
        X = scaler_X.transform(X_df)
        y = scaler_y.fit_transform(y_df.values.reshape(-1, 1)).flatten()
        years = X_df.index.astype(int)

        return X, y, years, X_df, scaler_X, scaler_y, sectors, macro_rates, events, max(years), y_df, expected_features, df

    except Exception as e:
        error_log.append(f"Error loading file: {str(e)}")
        st.error(f"Error loading file: {str(e)}")
        raise

# File uploader
uploaded_file = st.file_uploader("Upload your updated dataset (CSV, optional)", type=["csv"])
if uploaded_file:
    st.write("### Preview of the Uploaded CSV File")
    try:
        uploaded_file.seek(0)
        df_preview = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        st.write(df_preview)
        
        if st.button("Add a New Row"):
            year_columns = [col for col in df_preview.columns if col != 'année' and col.isdigit()]
            max_year = max([int(col) for col in year_columns]) if year_columns else 2023
            new_year = max_year + 1
            new_row = pd.DataFrame({col: ['gross domestic product gdp' if col == 'année' else 0.0] for col in df_preview.columns})
            if str(new_year) not in df_preview.columns:
                new_row[str(new_year)] = 0.0
            st.write(f"### Add Data for Year {new_year}")
            edited_row = st.data_editor(new_row, num_rows="dynamic")
            
            if st.button("Save the New Row"):
                for col in df_preview.columns:
                    if col not in edited_row.columns:
                        edited_row[col] = 0.0
                if str(new_year) not in df_preview.columns:
                    df_preview[str(new_year)] = 0.0
                df_updated = pd.concat([df_preview, edited_row], ignore_index=True)
                output_file = "updated_VA-2015-2023P.csv"
                df_updated.to_csv(output_file, sep=';', index=False, encoding='utf-8')
                st.success(f"New row saved to '{output_file}'.")
                with open(output_file, 'rb') as f:
                    uploaded_file = f
                uploaded_file.seek(0)
    except Exception as e:
        error_log.append(f"Error reading uploaded file for preview: {str(e)}")
        st.error(f"Error reading uploaded file: {str(e)}")
        st.stop()

# Load data
try:
    X, y, years, X_df, scaler_X, scaler_y, sectors, macro_rates, events, last_year, y_df, expected_features, df = load_and_preprocess(uploaded_file)
except (ValueError, FileNotFoundError, KeyError) as e:
    st.error(str(e))
    st.stop()

st.write(f"**Last Available Year in Data:** {last_year}")
st.write(f"**Number of Features in X_df:** {X_df.shape[1]} (expected: {len(expected_features)})")

# Cache model structure
@st.cache_resource(show_spinner=False)
def get_model_structure(model_type):
    if model_type == "Ridge":
        return Pipeline([
            ('feature_selection', SelectKBest(score_func=f_regression)),
            ('ridge', Ridge())
        ])
    elif model_type == "ElasticNet":
        return Pipeline([
            ('feature_selection', SelectKBest(score_func=f_regression)),
            ('elasticnet', ElasticNet())
        ])
    elif model_type == "Huber":
        return Pipeline([
            ('feature_selection', SelectKBest(score_func=f_regression)),
            ('huber', HuberRegressor(max_iter=1000))
        ])

# Define models
loo = LeaveOneOut()

ridge_params = {
    'ridge__alpha': np.logspace(-2, 3, 50),
    'feature_selection__k': [5, 10, 15, 20]
}
ridge_cv = RandomizedSearchCV(get_model_structure("Ridge"), ridge_params, cv=loo, scoring='neg_mean_absolute_error', n_iter=20, random_state=42)

elasticnet_params = {
    'elasticnet__alpha': np.logspace(-2, 3, 50),
    'elasticnet__l1_ratio': np.linspace(0.1, 0.9, 9),
    'feature_selection__k': [5, 10, 15, 20]
}
elasticnet_cv = RandomizedSearchCV(get_model_structure("ElasticNet"), elasticnet_params, cv=loo, scoring='neg_mean_absolute_error', n_iter=20, random_state=42)

huber_params = {
    'huber__epsilon': np.linspace(1.1, 2.0, 10),
    'huber__alpha': np.logspace(-4, 1, 20),
    'feature_selection__k': [5, 10, 15, 20]
}
huber_cv = RandomizedSearchCV(get_model_structure("Huber"), huber_params, cv=loo, scoring='neg_mean_absolute_error', n_iter=20, random_state=42)

# Evaluation and interpretation function
def interpret_results(model_name, train_mae, test_mae, train_r2, test_r2):
    rel_error = test_mae / np.mean(scaler_y.inverse_transform(y.reshape(-1, 1)))
    st.markdown("#### 💡 Interpretation")
    st.write(f"**Test R²:** {test_r2:.4f} — indicates generalization quality.")
    st.write(f"**Absolute MAE:** {test_mae:.0f} — for an average GDP ~{np.mean(scaler_y.inverse_transform(y.reshape(-1, 1))):,.0f}, i.e., a relative error of about **{rel_error*100:.1f}%**.")
    diff_r2 = train_r2 - test_r2
    if diff_r2 > 0.15:
        st.error("⚠️ Large gap between train and test R² → possible overfitting.")
    else:
        st.success("✅ No obvious signs of overfitting.")

    st.markdown("#### ✅ Conclusion")
    if test_r2 >= 0.96 and rel_error < 0.03:
        st.write(f"✔️ **{model_name} performs excellently.**")
        st.write("- Can be used as a benchmark.")
        st.write("- Highly reliable for GDP forecasting.")
    elif test_r2 >= 0.90:
        st.write(f"✔️ **{model_name} is a good model,** but could be improved.")
    else:
        st.write(f"❌ **{model_name} shows limitations.** Consider another method or further tuning.")

def evaluate_model(model_cv, X, y, model_name):
    model_cv.fit(X, y)
    train_pred = model_cv.predict(X)
    train_pred_unscaled = scaler_y.inverse_transform(train_pred.reshape(-1, 1)).flatten()
    y_unscaled = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
    train_mae = mean_absolute_error(y_unscaled, train_pred_unscaled)
    train_r2 = r2_score(y_unscaled, train_pred_unscaled)

    preds_test = []
    for tr, te in loo.split(X):
        best_model = model_cv.best_estimator_
        best_model.fit(X[tr], y[tr])
        preds_test.append(best_model.predict(X[te])[0])

    test_pred_unscaled = scaler_y.inverse_transform(np.array(preds_test).reshape(-1, 1)).flatten()
    test_mae = mean_absolute_error(y_unscaled, test_pred_unscaled)
    test_r2 = r2_score(y_unscaled, test_pred_unscaled)

    st.markdown(f"### 🔍 Results for **{model_name}**")
    st.write(f"Train MAE: {train_mae:.2f}, Test MAE (LeaveOneOut): {test_mae:.2f}")
    st.write(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    st.write(f"Best hyperparameters: {model_cv.best_params_}")

    interpret_results(model_name, train_mae, test_mae, train_r2, test_r2)
    return test_mae, test_r2, model_cv

# Run models
st.header("📊 Model Diagnostics and Interpretation")
results = []
models = {}
test_maes = {}

# Check if models need to be trained
train_models = True
if "last_input" in st.session_state and st.session_state.last_input == uploaded_file:
    if "trained_models" in st.session_state and "test_maes" in st.session_state:
        models = st.session_state.trained_models
        test_maes = st.session_state.test_maes
        results = st.session_state.results
        train_models = False
        st.write("Using results from previously trained models.")

# Train models if needed
if train_models:
    for model, name in [(ridge_cv, "Ridge"), (elasticnet_cv, "ElasticNet"), (huber_cv, "Huber")]:
        with st.spinner(f"Training {name}..."):
            mae, r2, trained_model = evaluate_model(model, X, y, name)
            results.append({
                'Model': name,
                'CV MAE': mae,
                'Train R²': r2_score(scaler_y.inverse_transform(y.reshape(-1, 1)), scaler_y.inverse_transform(model.predict(X).reshape(-1, 1)))
            })
            models[name] = trained_model
            test_maes[name] = mae
    st.session_state.trained_models = models
    st.session_state.test_maes = test_maes
    st.session_state.results = results
    st.session_state.last_input = uploaded_file

# Check if test_maes is empty
if not test_maes:
    st.error("No models were trained. Please check the input data or reset the session.")
    st.stop()

# Select best model
best_model_name = min(test_maes, key=test_maes.get)
best_model = models[best_model_name].best_estimator_
st.markdown(f"### 🏆 Selected Model: **{best_model_name}**")
st.write(f"The model **{best_model_name}** was chosen because it has the lowest MAE: {test_maes[best_model_name]:.2f}")

# Verify selected model
st.header("🔎 Verification of the Selected Model")
st.markdown("#### 1. Data Integrity Check")
if X_df.isna().any().any():
    error_log.append("Missing values detected in X_df.")
    st.error("Missing values in input data. Replacing with 0.")
    X_df = X_df.fillna(0)
if y_df.isna().any().any():
    error_log.append("Missing values detected in y_df.")
    st.warning("Missing values in target data. Replacing with mean.")
    y_df = y_df.fillna(y_df.mean())
if y_df.empty or y_df.shape[0] == 0:
    error_log.append("y_df is empty or has no rows.")
    st.error("Target data (y_df) is empty. Stopping the program.")
    st.stop()
st.success(f"No missing values in data after preprocessing. y_df shape: {y_df.shape}")

st.markdown("#### 2. Test Set Verification")
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
best_model.fit(X_train, y_train)
y_pred_test = best_model.predict(X_test)
y_pred_test_unscaled = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
test_mae = mean_absolute_error(y_test_unscaled, y_pred_test_unscaled)
test_r2 = r2_score(y_test_unscaled, y_pred_test_unscaled)
st.write(f"Test Set MAE: {test_mae:.2f}")
st.write(f"Test Set R²: {test_r2:.4f}")
if test_mae > 1.5 * test_maes[best_model_name]:
    error_log.append(f"Test set MAE ({test_mae:.2f}) significantly higher than CV MAE ({test_maes[best_model_name]:.2f}).")
    st.warning("Test set performance worse than expected.")

st.markdown("#### 3. Residual Analysis")
residuals = y_test_unscaled - y_pred_test_unscaled
fig_residuals = px.scatter(x=years[train_size:], y=residuals, title="Residuals on Test Set",
                           labels={'x': 'Year', 'y': 'Residuals (million TND)'}, color_discrete_sequence=['#FF6B6B'])
fig_residuals.add_hline(y=0, line_dash="dash", line_color="black")
st.plotly_chart(fig_residuals)
if np.abs(residuals).mean() > test_maes[best_model_name]:
    error_log.append(f"Average residuals ({np.abs(residuals).mean():.2f}) are high compared to CV MAE ({test_maes[best_model_name]:.2f}).")
    st.warning("Residuals show high average error, indicating potential underperformance.")

st.markdown("#### 4. Prediction Intervals")
n_bootstraps = 100
bootstrap_preds = []
for _ in range(n_bootstraps):
    indices = np.random.choice(len(X_train), len(X_train), replace=True)
    best_model.fit(X_train[indices], y_train[indices])
    pred = best_model.predict(X_test)
    bootstrap_preds.append(scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten())
bootstrap_preds = np.array(bootstrap_preds)
lower_bound = np.percentile(bootstrap_preds, 2.5, axis=0)
upper_bound = np.percentile(bootstrap_preds, 97.5, axis=0)
st.write("95% Prediction Intervals for Test Set:")
for i, (lower, upper, actual) in enumerate(zip(lower_bound, upper_bound, y_test_unscaled)):
    st.write(f"Year {years[train_size+i]}: Predicted = {y_pred_test_unscaled[i]:,.0f}, Interval = [{lower:,.0f}, {upper:,.0f}], Actual = {actual:,.0f}")

# Prediction for 2024
if st.button("🔮 Predict GDP for 2024"):
    with st.spinner("Training and predicting..."):
        target_year = last_year + 1
        historical_df = pd.DataFrame({'Year': years, 'GDP': scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()})
        pred_df = pd.DataFrame({'Year': [target_year], 'GDP': [0.0]})
        combined_df = pd.concat([historical_df, pred_df], ignore_index=True)

        feature_vector = pd.DataFrame(index=[0], columns=expected_features).fillna(0.0)
        base_year_data = X_df.loc[last_year] if last_year in X_df.index else X_df.iloc[-3:].mean()

        recent_data = X_df[expected_features].tail(3)
        growth_rates = {}
        for col in sectors + macro_rates:
            if col in recent_data.columns:
                year_growth = recent_data[col].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
                growth_rates[col] = year_growth.mean() * 100 if not year_growth.empty else 0.0
            else:
                growth_rates[col] = 0.0
                error_log.append(f"Growth rate for '{col}' not calculated (column missing). Using 0.")
        
        for event in events:
            if event in recent_data.columns:
                growth_rates[event] = recent_data[event].mean() if not recent_data[event].empty else 0
            else:
                growth_rates[event] = 0
                error_log.append(f"Value for '{event}' not found. Using 0.")

        for sector in sectors:
            try:
                if sector not in X_df.columns:
                    error_log.append(f"Error for {sector} ({target_year}): not found in X_df. Using 0.")
                    feature_vector[sector] = 0.0
                else:
                    feature_vector[sector] = base_year_data[sector] * (1 + growth_rates[sector] / 100)
            except Exception as e:
                error_log.append(f"Error for {sector} ({target_year}): {str(e)}. Using 0.")
                feature_vector[sector] = 0.0

        for rate in macro_rates:
            try:
                if rate not in X_df.columns:
                    error_log.append(f"Error for {rate} ({target_year}): not found in X_df. Using 0.")
                    feature_vector[rate] = 0.0
                else:
                    feature_vector[rate] = base_year_data[rate] * (1 + growth_rates[rate] / 100)
            except Exception as e:
                error_log.append(f"Error for {rate} ({target_year}): {str(e)}. Using 0.")
                feature_vector[rate] = 0.0

        for event in events:
            try:
                if event in X_df.columns:
                    feature_vector[event] = growth_rates[event]
                else:
                    error_log.append(f"Error for {event} ({target_year}): not found in X_df. Using 0.")
                    feature_vector[event] = 0
            except Exception as e:
                error_log.append(f"Error for {event} ({target_year}): {str(e)}. Using 0.")
                feature_vector[event] = 0

        for col in expected_features:
            if col not in sectors + macro_rates + events:
                if col.endswith('_lag1'):
                    base_col = col.replace('_lag1', '')
                    if base_col in feature_vector.columns:
                        feature_vector[col] = base_year_data.get(base_col, X_df[base_col].mean() if base_col in X_df.columns else 0.0)
                    else:
                        feature_vector[col] = base_year_data.get(col, X_df[col].mean() if col in X_df.columns else 0.0)
                    if feature_vector[col].iloc[0] == 0.0:
                        error_log.append(f"Lagged feature '{col}' for {target_year} set to 0 (missing data).")
                else:
                    feature_vector[col] = base_year_data.get(col, X_df[col].mean() if col in X_df.columns else 0.0)
                    if feature_vector[col].iloc[0] == 0.0:
                        error_log.append(f"Feature '{col}' for {target_year} set to 0 (missing data).")

        if feature_vector.isna().any().any():
            error_log.append(f"NaN values for {target_year}: {feature_vector.columns[feature_vector.isna().any()].tolist()}. Replacing with 0.")
            feature_vector = feature_vector.fillna(0.0)

        feature_vector = feature_vector[expected_features]
        error_log.append(f"Feature vector for {target_year}: {list(feature_vector.columns)} (count: {len(feature_vector.columns)})")
        X_new = scaler_X.transform(feature_vector)

        predicted_gdp = float(scaler_y.inverse_transform(best_model.predict(X_new).reshape(-1, 1))[0])
        combined_df.loc[combined_df['Year'] == target_year, 'GDP'] = predicted_gdp

        st.markdown("### 📈 Prediction Result")
        st.write(f"**Model Used:** {best_model_name}")
        st.write(f"**Predicted GDP for {target_year}:** {predicted_gdp:,.0f} million TND")

        fig = px.line(combined_df, x='Year', y='GDP', title=f'Historical and Predicted GDP for {target_year} (based on {best_model_name})',
                      markers=True, color_discrete_sequence=['#45B7D1'])
        fig.add_scatter(x=[target_year], y=[predicted_gdp], mode='markers', marker=dict(size=10, color='red'), name=f'Prediction {target_year}')
        st.plotly_chart(fig)

        st.markdown("### 🧠 Prediction Explanation with SHAP")
        st.write(f"The following plots explain how each feature contributes to the GDP prediction for {target_year}.")

        best_model.fit(X, y)
        feature_vector_for_shap = X_new
        error_log.append(f"Shape of feature_vector_for_shap: {feature_vector_for_shap.shape}")
        background_data = scaler_X.transform(X_df[expected_features])
        error_log.append(f"Shape of background_data: {background_data.shape}")

        try:
            if best_model_name in ["Ridge", "ElasticNet"]:
                explainer = shap.LinearExplainer(
                    best_model,
                    background_data,
                    feature_names=expected_features
                )
            else:  # Huber
                explainer = shap.KernelExplainer(
                    best_model.predict,
                    background_data,
                    feature_names=expected_features
                )

            shap_values = explainer.shap_values(feature_vector_for_shap)
            error_log.append(f"Shape of shap_values: {np.array(shap_values).shape}")

            st.markdown("#### 📊 Global Feature Importance (Summary Plot)")
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, feature_vector_for_shap, feature_names=expected_features, show=False)
            st.pyplot(plt)
            plt.close()

            st.markdown("#### 📉 Dependence Plot for gdp_lag1")
            plt.figure(figsize=(10, 6))
            shap.dependence_plot("gdp_lag1", shap_values, feature_vector_for_shap, feature_names=expected_features, show=False)
            st.pyplot(plt)
            plt.close()

            st.markdown(f"#### 📈 Feature Contributions for {target_year} (Force Plot)")
            plt.figure(figsize=(10, 4))
            shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                feature_vector_for_shap[0],
                feature_names=expected_features,
                matplotlib=True,
                show=False
            )
            st.pyplot(plt)
            plt.close()

            st.markdown(f"#### 📊 Feature Importance for {target_year}")
            plt.figure(figsize=(10, 6))
            shap.bar_plot(shap_values[0], feature_names=expected_features, max_display=10, show=False)
            st.pyplot(plt)
            plt.close()

        except Exception as e:
            error_log.append(f"Error computing SHAP: {str(e)}")
            st.error(f"Unable to generate SHAP explanations: {str(e)}. Please check the data.")

        st.markdown("#### 📈 Historical vs Predicted GDP")
        y_pred_historical = best_model.predict(X)
        y_pred_historical_unscaled = scaler_y.inverse_transform(y_pred_historical.reshape(-1, 1)).flatten()
        y_historical_unscaled = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
        historical_df = pd.DataFrame({
            'Year': years,
            'Actual GDP': y_historical_unscaled,
            'Predicted GDP': y_pred_historical_unscaled
        })
        pred_df = pd.DataFrame({
            'Year': [target_year],
            'Actual GDP': [np.nan],
            'Predicted GDP': [predicted_gdp]
        })
        combined_df = pd.concat([historical_df, pred_df], ignore_index=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=combined_df['Year'], y=combined_df['Actual GDP'], mode='lines+markers', name='Actual GDP', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=combined_df['Year'], y=combined_df['Predicted GDP'], mode='lines+markers', name='Predicted GDP', line=dict(color='red', dash='dash')))
        fig.update_layout(title=f'Historical vs Predicted GDP (incl. {target_year})', xaxis_title='Year', yaxis_title='GDP (million TND)')
        st.plotly_chart(fig)

        st.info(f"🧪 Prediction based on the {best_model_name} model with the lowest MAE, using trends extrapolated from the last 3 years.")

        show_errors = st.checkbox("Show Log", value=True)
        if show_errors and error_log:
            st.markdown("### Informational Log")
            for error in error_log:
                st.write(error)