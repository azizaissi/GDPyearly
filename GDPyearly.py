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
current_date_time = cet.localize(datetime(2025, 7, 26, 23, 49))  # Updated to 11:49 PM CET
st.write(f"**Current Date and Time:** {current_date_time.strftime('%d/%m/%Y %H:%M %Z')}")

# Set random seed
random.seed(42)
np.random.seed(42)

# Initialize error log
error_log = []

# Normalize string function
def normalize_name(name):
    if pd.isna(name) or not isinstance(name, str):
        error_log.append(f"Valeur non textuelle ou NaN : {name}. Remplacement par 'inconnu'.")
        return "inconnu"
    original_name = name
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8').strip()
    name = re.sub(r"['’´]+", "'", name)
    name = re.sub(r'\s+', ' ', name).lower()
    name = name.replace("d'autre produits", "d'autres produits")
    name = name.replace("de lhabillement", "de l'habillement")
    name = name.replace("crise sociale", "Crise sociale")
    if name.startswith("impots nets de subventions") or name.startswith("impôts nets de subventions"):
        name = "impots nets de subventions sur les produits"
        error_log.append(f"Normalisé '{original_name}' en 'impots nets de subventions sur les produits'.")
    error_log.append(f"Normalisation : '{original_name}' -> '{name}'")
    return name

# Load and preprocess data
@st.cache_data
def load_and_preprocess(uploaded_file=None):
    try:
        if uploaded_file:
            uploaded_file.seek(0)
            raw_content = uploaded_file.read().decode('utf-8')
            if not raw_content.strip():
                error_log.append("Le fichier uploadé est vide.")
                st.error("The uploaded file is empty. Please check the file.")
                raise ValueError("Fichier vide.")
            error_log.append(f"Contenu brut du fichier uploadé : {raw_content[:200]}...")
            
            for sep in [';', ',', '\t']:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=sep, encoding='utf-8')
                    if not df.empty and 'année' in df.columns:
                        error_log.append(f"Fichier chargé avec séparateur '{sep}'.")
                        break
                except Exception as e:
                    error_log.append(f"Échec de lecture avec séparateur '{sep}': {str(e)}")
            else:
                error_log.append("Impossible de lire le fichier avec les séparateurs testés (; , \\t).")
                st.error("Unable to read the CSV file. Check the format and separator.")
                raise ValueError("Format CSV invalide ou séparateur incorrect.")
        else:
            default_file = "VA-2015-2023P.csv"
            if not os.path.exists(default_file):
                error_log.append(f"Fichier '{default_file}' introuvable.")
                st.error(f"File '{default_file}' not found. Check the file path.")
                raise FileNotFoundError(f"Fichier '{default_file}' introuvable.")
            df = pd.read_csv(default_file, sep=';', encoding='utf-8')
            error_log.append(f"Fichier chargé comme CSV avec séparateur ';'.")

        if df.empty or len(df.columns) == 0:
            error_log.append("Le fichier CSV ne contient aucune colonne valide.")
            st.error("The CSV file contains no valid columns. Check the file content.")
            raise ValueError("Aucune colonne dans le fichier CSV.")
        if 'année' not in df.columns:
            error_log.append(f"Colonne 'année' absente. Colonnes trouvées : {df.columns.tolist()}")
            st.error(f"The 'année' column is required. Found columns: {df.columns.tolist()}")
            raise ValueError("Colonne 'année' manquante.")

        df = df.rename(columns={'année': 'Secteur'})
        error_log.append(f"Secteurs bruts dans le CSV : {df['Secteur'].tolist()}")
        df['Secteur'] = df['Secteur'].apply(normalize_name)
        error_log.append(f"Secteurs après normalisation : {df['Secteur'].tolist()}")

        for col in df.columns[1:]:
            df[col] = df[col].astype(str).str.replace(' ', '').str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        sectors = [
            "agriculture, sylviculture et peche",
            "extraction petrole et gaz naturel",
            "extraction des produits miniers",
            "industries agro-alimentaires",
            "industrie du textile, de l'habillement et du cuir",
            "raffinage du petrole",
            "industries chimiques",
            "industrie d'autres produits mineraux non metalliques",
            "industries mecaniques et electriques",
            "industries diverses",
            "production et distribution de l'electricite et gaz",
            "production et distribution d'eau et gestion des dechets",
            "construction",
            "commerce et reparation",
            "transport et entreposage",
            "hebergement et restauration",
            "information et communication",
            "activites financieres et d'assurances",
            "administration publique et defense",
            "enseignement",
            "sante humaine et action sociale",
            "autres services marchands",
            "autres activites des menages",
            "activites des organisations associatives"
        ]
        macro_keywords = [
            "taux de chomage", "taux d'inflation", "taux d'interet", "dette publique", "pression fiscale",
            "politique monetaire internationale", "tensions geopolitiques regionales", "prix matieres premieres",
            "secheresse et desastre climatique", "pandemies", "Crise sociale",
            "impots nets de subventions sur les produits"
        ]
        macro_rates = ["taux de chomage", "taux d'inflation", "taux d'interet", "dette publique", "pression fiscale"]
        events = [
            "politique monetaire internationale", "tensions geopolitiques regionales", "prix matieres premieres",
            "secheresse et desastre climatique", "pandemies", "Crise sociale"
        ]

        if 'f' in df['Secteur'].values:
            error_log.append("Ligne 'f' détectée dans les secteurs. Elle sera exclue.")
            df = df[df['Secteur'] != 'f']

        if not df['Secteur'].str.contains("produit interieur brut pib", case=False).any():
            st.error(f"No GDP data found. Available sectors: {df['Secteur'].tolist()}")
            error_log.append("Aucune donnée PIB trouvée dans le fichier.")
            raise ValueError("Données PIB manquantes.")

        impots_key = "impots nets de subventions sur les produits"
        df_macro = df[df['Secteur'].isin(macro_keywords)].copy()
        df_pib = df[df['Secteur'] == "produit interieur brut pib"].copy()
        df_secteurs = df[df['Secteur'].isin(sectors)].copy()
        df_secteurs = df_secteurs[df_secteurs['Secteur'] != impots_key]

        if impots_key not in df_macro['Secteur'].values:
            error_log.append(f"Erreur : '{impots_key}' non trouvé dans df_macro.")
            st.error(f"Error: '{impots_key}' not found in macro data.")
            raise ValueError(f"'{impots_key}' manquant dans df_macro.")
        if impots_key in df_secteurs['Secteur'].values:
            error_log.append(f"Erreur : '{impots_key}' trouvé dans df_secteurs après exclusion.")
            st.error(f"Error: '{impots_key}' found in sector data after exclusion.")
            raise ValueError(f"'{impots_key}' trouvé dans df_secteurs après exclusion.")

        error_log.append(f"Secteurs dans df_secteurs : {df_secteurs['Secteur'].tolist()}")
        error_log.append(f"Macros dans df_macro : {df_macro['Secteur'].tolist()}")

        if df_pib.empty:
            st.error(f"No GDP data found. Available sectors: {df['Secteur'].tolist()}")
            error_log.append("Aucune donnée PIB trouvée dans le fichier.")
            raise ValueError("Données PIB manquantes.")

        missing_sectors = [s for s in sectors if s not in df['Secteur'].values]
        missing_macro = [m for m in macro_keywords if m not in df['Secteur'].values]
        if missing_sectors:
            st.warning(f"Missing sectors: {missing_sectors}. Using average of available sectors.")
            error_log.append(f"Secteurs manquants : {missing_sectors}")
        if missing_macro:
            st.warning(f"Missing macro variables: {missing_macro}. Using default values (0).")
            error_log.append(f"Macros manquants : {missing_macro}")

        df_macro.set_index("Secteur", inplace=True)
        df_pib.set_index("Secteur", inplace=True)
        df_secteurs.set_index("Secteur", inplace=True)

        df_macro_T = df_macro.transpose()
        df_secteurs_T = df_secteurs.transpose()
        df_pib_T = df_pib.transpose()

        X_df = pd.concat([df_secteurs_T, df_macro_T[macro_rates + events]], axis=1).dropna()
        y_df = df_pib_T.loc[X_df.index]

        error_log.append(f"Colonnes dans X_df après concaténation : {list(X_df.columns)}")

        if y_df.empty:
            st.error(f"y_df empty after alignment with X_df. X_df indices: {X_df.index.tolist()}. df_pib_T indices: {df_pib_T.index.tolist()}")
            error_log.append("y_df vide après alignement.")
            raise ValueError("Données PIB vides après prétraitement.")

        key_sectors = [
            "agriculture, sylviculture et peche", "industries mecaniques et electriques",
            "hebergement et restauration", "information et communication",
            "activites financieres et d'assurances"
        ]
        for sector in key_sectors:
            if sector in X_df.columns:
                X_df[f"{sector}_lag1"] = X_df[sector].shift(1).fillna(X_df[sector].mean())
            else:
                X_df[f"{sector}_lag1"] = X_df[sectors].mean(axis=1).shift(1).fillna(X_df[sectors].mean().mean()) if sectors else 0
                error_log.append(f"Feature décalée '{sector}_lag1' ajoutée avec moyenne des secteurs car '{sector}' est absent.")

        for rate in macro_rates:
            if rate in X_df.columns:
                X_df[f"{rate}_lag1"] = X_df[rate].shift(1).fillna(X_df[rate].mean())
            else:
                X_df[f"{rate}_lag1"] = 0
                error_log.append(f"Feature décalée '{rate}_lag1' ajoutée avec valeur 0 car '{rate}' est absent.")

        X_df['gdp_lag1'] = y_df.shift(1).fillna(y_df.mean())

        expected_features = sectors + macro_rates + events + [f"{s}_lag1" for s in key_sectors] + [f"{r}_lag1" for r in macro_rates] + ['gdp_lag1']
        error_log.append(f"Colonnes attendues dans X_df : {expected_features} (nombre: {len(expected_features)})")

        missing_cols = [col for col in expected_features if col not in X_df.columns]
        extra_cols = [col for col in X_df.columns if col not in expected_features]
        if missing_cols:
            existing_cols = [col for col in sectors + macro_rates + events if col in X_df.columns]
            for col in missing_cols:
                if col in sectors and existing_cols:
                    X_df[col] = X_df[existing_cols].mean(axis=1)
                    error_log.append(f"Feature manquante '{col}' ajoutée avec la moyenne des secteurs disponibles.")
                elif col.endswith('_lag1') and col.replace('_lag1', '') in X_df.columns:
                    X_df[col] = X_df[col.replace('_lag1', '')].shift(1).fillna(X_df[col.replace('_lag1', '')].mean())
                    error_log.append(f"Feature manquante '{col}' ajoutée avec décalage.")
                else:
                    X_df[col] = 0
                    error_log.append(f"Feature manquante '{col}' ajoutée avec valeur 0.")
        if extra_cols:
            st.warning(f"Extra columns in X_df: {extra_cols}")
            error_log.append(f"Colonnes supplémentaires dans X_df : {extra_cols}")
            X_df = X_df.drop(columns=extra_cols, errors='ignore')

        error_log.append(f"Colonnes dans X_df après ajout des features manquantes : {list(X_df.columns)}")
        error_log.append(f"Nombre de colonnes dans X_df : {X_df.shape[1]} (attendu : {len(expected_features)})")

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_df = X_df[expected_features]
        scaler_X.fit(X_df)
        error_log.append(f"Scaler_X ajusté sur {scaler_X.n_features_in_} features")
        X = scaler_X.transform(X_df)
        y = scaler_y.fit_transform(y_df.values.reshape(-1, 1)).flatten()
        years = X_df.index.astype(int)

        return X, y, years, X_df, scaler_X, scaler_y, sectors, macro_rates, events, max(years), y_df, expected_features, df

    except Exception as e:
        error_log.append(f"Erreur lors du chargement du fichier : {str(e)}")
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
            new_row = pd.DataFrame({col: ['produit interieur brut pib' if col == 'année' else 0.0] for col in df_preview.columns})
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
        error_log.append(f"Erreur lors de la lecture du fichier uploadé pour l'aperçu : {str(e)}")
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

train_models = True
if "last_input" in st.session_state and st.session_state.last_input == uploaded_file:
    if "trained_models" in st.session_state and "test_maes" in st.session_state:
        models = st.session_state.trained_models
        test_maes = st.session_state.test_maes
        results = st.session_state.results
        train_models = False
        st.write("Using results from previously trained models.")

if train_models:
    for model, name in [(ridge_cv, "Ridge"), (elasticnet_cv, "ElasticNet"), (huber_cv, "Huber")]:
        with st.spinner(f"Training {name}..."):
            mae, r2, trained_model = evaluate_model(model, X, y, name)
            results.append({
                'Modèle': name,
                'CV MAE': mae,
                'Train R²': r2_score(scaler_y.inverse_transform(y.reshape(-1, 1)), scaler_y.inverse_transform(model.predict(X).reshape(-1, 1)))
            })
            models[name] = trained_model
            test_maes[name] = mae
    st.session_state.trained_models = models
    st.session_state.test_maes = test_maes
    st.session_state.results = results
    st.session_state.last_input = uploaded_file

if not test_maes:
    st.error("No models were trained. Please check the input data or reset the session.")
    st.stop()

best_model_name = min(test_maes, key=test_maes.get)
best_model = models[best_model_name].best_estimator_
st.markdown(f"### 🏆 Selected Model: **{best_model_name}**")
st.write(f"The model **{best_model_name}** was chosen because it has the lowest MAE: {test_maes[best_model_name]:.2f}")

st.header("🔎 Verification of the Selected Model")
st.markdown("#### 1. Data Integrity Check")
if X_df.isna().any().any():
    error_log.append("Valeurs manquantes détectées dans X_df.")
    st.error("Missing values in input data. Replacing with 0.")
    X_df = X_df.fillna(0)
if y_df.isna().any().any():
    error_log.append("Valeurs manquantes détectées dans y_df.")
    st.warning("Missing values in target data. Replacing with mean.")
    y_df = y_df.fillna(y_df.mean())
if y_df.empty or y_df.shape[0] == 0:
    error_log.append("y_df est vide ou n'a aucune ligne.")
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
    error_log.append(f"MAE sur l'ensemble de test ({test_mae:.2f}) significativement plus élevé que le MAE CV ({test_maes[best_model_name]:.2f}).")
    st.warning("Test set performance worse than expected.")

st.markdown("#### 3. Residual Analysis")
residuals = y_test_unscaled - y_pred_test_unscaled
fig_residuals = px.scatter(x=years[train_size:], y=residuals, title="Residuals on Test Set",
                           labels={'x': 'Year', 'y': 'Residuals (million TND)'}, color_discrete_sequence=['#FF6B6B'])
fig_residuals.add_hline(y=0, line_dash="dash", line_color="black")
st.plotly_chart(fig_residuals)
if np.abs(residuals).mean() > test_maes[best_model_name]:
    error_log.append(f"Les résidus moyens ({np.abs(residuals).mean():.2f}) sont élevés par rapport au MAE CV ({test_maes[best_model_name]:.2f}).")
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
                error_log.append(f"Taux de croissance pour '{col}' non calculé (colonne absente). Utilisation de 0.")
        
        for event in events:
            if event in recent_data.columns:
                growth_rates[event] = recent_data[event].mean() if not recent_data[event].empty else 0
            else:
                growth_rates[event] = 0
                error_log.append(f"Valeur pour '{event}' non trouvée. Utilisation de 0.")

        for sector in sectors:
            try:
                if sector not in X_df.columns:
                    error_log.append(f"Erreur pour {sector} ({target_year}): non trouvé dans X_df. Utilisation de 0.")
                    feature_vector[sector] = 0.0
                else:
                    feature_vector[sector] = base_year_data[sector] * (1 + growth_rates[sector] / 100)
            except Exception as e:
                error_log.append(f"Erreur pour {sector} ({target_year}): {str(e)}. Utilisation de 0.")
                feature_vector[sector] = 0.0

        for rate in macro_rates:
            try:
                if rate not in X_df.columns:
                    error_log.append(f"Erreur pour {rate} ({target_year}): non trouvé dans X_df. Utilisation de 0.")
                    feature_vector[rate] = 0.0
                else:
                    feature_vector[rate] = base_year_data[rate] * (1 + growth_rates[rate] / 100)
            except Exception as e:
                error_log.append(f"Erreur pour {rate} ({target_year}): {str(e)}. Utilisation de 0.")
                feature_vector[rate] = 0.0

        for event in events:
            try:
                if event in X_df.columns:
                    feature_vector[event] = growth_rates[event]
                else:
                    error_log.append(f"Erreur pour {event} ({target_year}): non trouvé dans X_df. Utilisation de 0.")
                    feature_vector[event] = 0
            except Exception as e:
                error_log.append(f"Erreur pour {event} ({target_year}): {str(e)}. Utilisation de 0.")
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
                        error_log.append(f"Feature décalée '{col}' pour {target_year} définie à 0 (données manquantes).")
                else:
                    feature_vector[col] = base_year_data.get(col, X_df[col].mean() if col in X_df.columns else 0.0)
                    if feature_vector[col].iloc[0] == 0.0:
                        error_log.append(f"Feature '{col}' pour {target_year} définie à 0 (données manquantes).")

        if feature_vector.isna().any().any():
            error_log.append(f"Valeurs NaN pour {target_year} : {feature_vector.columns[feature_vector.isna().any()].tolist()}. Remplacement par 0.")
            feature_vector = feature_vector.fillna(0.0)

        feature_vector = feature_vector[expected_features]
        error_log.append(f"Feature vector pour {target_year} : {list(feature_vector.columns)} (nombre: {len(feature_vector.columns)})")
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
        error_log.append(f"Shape de feature_vector_for_shap : {feature_vector_for_shap.shape}")
        background_data = scaler_X.transform(X_df[expected_features])
        error_log.append(f"Shape de background_data : {background_data.shape}")

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
            error_log.append(f"Shape de shap_values : {np.array(shap_values).shape}")

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
            error_log.append(f"Erreur lors du calcul SHAP : {str(e)}")
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
