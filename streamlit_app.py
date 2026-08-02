import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['text.usetex'] = False
import seaborn as sns
import pickle
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, log_loss)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE
import shap
import lime
import lime.lime_tabular
import streamlit.components.v1 as components
from sklearn.inspection import permutation_importance

# Import TensorFlow only when needed (lazy loading)

# Set page configuration
st.set_page_config(
    page_title="ASD Screening Prediction System",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Autism Spectrum Disorder (ASD) Screening Prediction System")
st.markdown("---")

# Sidebar for navigation
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "🤖 Model Training", "🔮 Make Prediction", "📊 Model Comparison"]
)

# Module-level constant for pretrained model storage
PRETRAINED_DIR = "pretrained_models"

# ==========================================
# UTILITY & XAI HELPER FUNCTIONS
# ==========================================

def generate_clinical_narrative(feature_names, input_values, shap_values, base_value, probability, model_name):
    """Generate human-readable clinical explanation based on SHAP feature attributions"""
    prob_percent = probability * 100
    is_positive = probability >= 0.5
    diagnosis_str = "🔴 High Risk of ASD (Positive)" if is_positive else "🟢 Low Risk of ASD (Negative)"
    
    feature_impacts = []
    for name, val, shap_val in zip(feature_names, input_values, shap_values):
        feature_impacts.append({
            'name': name,
            'value': val,
            'shap': shap_val,
            'abs_shap': abs(shap_val)
        })
    
    feature_impacts.sort(key=lambda x: x['abs_shap'], reverse=True)
    
    risk_increasing = [f for f in feature_impacts if f['shap'] > 0.005]
    risk_decreasing = [f for f in feature_impacts if f['shap'] < -0.005]
    
    narrative = f"""
#### 🩺 Clinical Decision Narrative ({model_name})

- **Diagnostic Risk Prediction:** **{diagnosis_str}** with a confidence score of **{prob_percent:.1f}%**.
- **Population Baseline Risk:** `{base_value * 100:.1f}%` (average baseline probability).
"""
    
    if risk_increasing:
        narrative += "\n##### ⚠️ Top Risk-Elevating Factors:\n"
        for item in risk_increasing[:5]:
            narrative += f"- **{item['name']}** (Score/Value: `{item['value']:.1f}`): Contributed **+{item['shap']*100:.2f}%** toward ASD risk.\n"
            
    if risk_decreasing:
        narrative += "\n##### 🟢 Top Protective / Low-Risk Factors:\n"
        for item in risk_decreasing[:5]:
            narrative += f"- **{item['name']}** (Score/Value: `{item['value']:.1f}`): Reduced ASD risk contribution by **{item['shap']*100:.2f}%**.\n"
            
    narrative += """
---
*SHAP (SHapley Additive exPlanations) computes exact game-theoretic contributions of each clinical parameter relative to expected population baseline values.*
"""
    return narrative

def compute_shap_explanation(model, input_scaled, feature_names, X_train_scaled, input_values):
    """Compute SHAP explanation object, values array, and baseline value"""
    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_obj = explainer(input_scaled)
            if len(shap_obj.shape) == 3:
                values = shap_obj.values[0, :, 1]
                base_val = shap_obj.base_values[0, 1]
            else:
                values = shap_obj.values[0]
                base_val = shap_obj.base_values[0]
        else:
            bg_sample = shap.sample(X_train_scaled, 40, random_state=42)
            explainer = shap.KernelExplainer(model.predict_proba, bg_sample)
            shap_vals = explainer.shap_values(input_scaled)
            if isinstance(shap_vals, list):
                values = shap_vals[1][0]
                base_val = explainer.expected_value[1]
            elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
                values = shap_vals[0, :, 1]
                base_val = explainer.expected_value[1]
            else:
                values = shap_vals[0]
                base_val = explainer.expected_value
                
        if hasattr(base_val, "item"):
            base_val = float(base_val.item())
        elif isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[0])
        else:
            base_val = float(base_val)
            
        explanation = shap.Explanation(
            values=np.array(values, dtype=float),
            base_values=base_val,
            data=np.array(input_values, dtype=float),
            feature_names=feature_names
        )
        return explanation, values, base_val
    except Exception as e:
        st.error(f"Error computing SHAP values: {e}")
        return None, None, None


def compute_lime_explanation(model, input_scaled, feature_names, X_train_scaled):
    """Compute LIME tabular instance explanation"""
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_scaled,
            feature_names=feature_names,
            class_names=['ASD Negative', 'ASD Positive'],
            mode='classification',
            random_state=42
        )
        exp = explainer.explain_instance(
            data_row=input_scaled[0],
            predict_fn=model.predict_proba,
            num_features=10
        )
        return exp
    except Exception as e:
        st.error(f"Error computing LIME explanation: {e}")
        return None


@st.cache_data
def load_data():
    """Load the CSV dataset - cached for performance"""
    df = pd.read_csv("Autism_Screening_Data_Combined.csv")
    return df

@st.cache_data
def prepare_data(df):
    """Preprocess the data - cached for performance"""
    df = df.drop_duplicates()
    df = df.fillna(df.mode().iloc[0])
    
    # Identify numeric and categorical features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Encode categorical variables
    le_dict = {}
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        le_dict[col] = le
    
    # Ensure all columns are numeric
    for col in df_encoded.columns:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
    
    df_encoded.fillna(df_encoded.mean(), inplace=True)
    
    return df_encoded, le_dict, numeric_cols, categorical_cols

def train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Train all ML models"""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            ccp_alpha=0.005,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=15,
            min_samples_leaf=6,
            max_features='sqrt',
            min_impurity_decrease=0.001,
            random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=51,
            weights='uniform',
            metric='minkowski',
            p=1
        ),
        "SVM (Poly)": SVC(kernel='poly', degree=2, C=0.1, gamma='scale', probability=True),
        "SVM (RBF)": SVC(kernel='rbf', C=0.1, gamma='scale', probability=True),
        "Naive Bayes": GaussianNB(var_smoothing=1e-8),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.7)
    }
    
    results = []
    roc_data = []
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        logloss = log_loss(y_test, y_prob)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        gap = train_acc - acc

        results.append([name, train_acc, acc, gap, prec, rec, specificity, f1, roc_auc, logloss])
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data.append((name, fpr, tpr, roc_auc))
    
    results_df = pd.DataFrame(results, columns=["Model", "Train Accuracy", "Accuracy", "Gap",
                                                "Precision", "Recall", "Specificity",
                                                "F1 Score", "ROC-AUC", "Log Loss"])
    
    return results_df, roc_data, trained_models

def train_ann(X_train_scaled, X_test_scaled, y_train, y_test):
    """Train ANN model using sklearn MLPClassifier"""
    ann = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42
    )

    ann.fit(X_train_scaled, y_train)

    y_prob_ann = ann.predict_proba(X_test_scaled)[:, 1]
    y_pred_ann = ann.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, ann.predict(X_train_scaled))

    acc = accuracy_score(y_test, y_pred_ann)
    prec = precision_score(y_test, y_pred_ann)
    rec = recall_score(y_test, y_pred_ann)
    f1 = f1_score(y_test, y_pred_ann)
    roc_auc = roc_auc_score(y_test, y_prob_ann)
    logloss = log_loss(y_test, y_prob_ann)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ann).ravel()
    specificity = tn / (tn + fp)
    gap = train_acc - acc

    fpr_ann, tpr_ann, _ = roc_curve(y_test, y_prob_ann)

    return ann, [train_acc, acc, gap, prec, rec, specificity, f1, roc_auc, logloss], (fpr_ann, tpr_ann, roc_auc), None

# ==========================================
# PAGE 1: HOME
# ==========================================
if page == "🏠 Home":
    st.subheader("Welcome to the ASD Screening Prediction System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        ### About This Application
        
        This application uses **Machine Learning** and **Artificial Neural Networks (ANN)** 
        to predict the likelihood of Autism Spectrum Disorder (ASD) based on screening data.
        
        **Features:**
        - Train multiple ML models
        - Compare model performance
        - Make predictions on new data
        - Visualize model performance metrics
        """)
    
    with col2:
        st.success("""
        ### ✅ Ready to Use!
        
        **Next Steps:**
        1. Click **🤖 Model Training** to train models
        2. Click **🔮 Make Prediction** to make predictions
        3. Click **📊 Model Comparison** to see results
        
        The app loads data on-demand for optimal performance.
        """)
    
    st.markdown("---")
    st.markdown("**💡 Tip:** Start with the Model Training tab to get predictions!")


# ==========================================
# PAGE 2: MODEL TRAINING
# ==========================================
elif page == "🤖 Model Training":

    st.subheader("🤖 Train Models")
    
    try:
        df = load_data()
        df_encoded, le_dict, numeric_cols, categorical_cols = prepare_data(df)
        
        # Split features and target
        X = df_encoded.iloc[:, :-1]
        y = df_encoded.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # ---- Mode selection: Train fresh or Load pretrained ----
        pretrained_exists = os.path.isdir(PRETRAINED_DIR) and os.path.isfile(
            os.path.join(PRETRAINED_DIR, "scaler.pkl")
        )
        
        if pretrained_exists:
            train_mode = st.radio(
                "🔧 Model Mode:",
                ["🚀 Train New Models", "📂 Load Pretrained Models"],
                horizontal=True,
                key="train_mode_radio"
            )
        else:
            train_mode = "🚀 Train New Models"
            st.info("💡 No saved models found yet. Train and save models to enable loading them later.")
        
        st.markdown("---")
        
        if train_mode == "📂 Load Pretrained Models":
            # ---- LOAD PRETRAINED ----
            st.subheader("📂 Load Previously Saved Models")
            
            saved_files = [f for f in os.listdir(PRETRAINED_DIR) if f.endswith(".pkl")]
            model_names_saved = [f.replace(".pkl", "") for f in saved_files if f != "scaler.pkl"]
            
            st.write(f"**Found {len(model_names_saved)} saved model(s):**")
            st.write(", ".join(f"`{n}`" for n in model_names_saved))
            
            if st.button("📂 Load Pretrained Models", key="load_pretrained_button"):
                with st.spinner("Loading saved models from disk..."):
                    try:
                        # Load scaler
                        with open(os.path.join(PRETRAINED_DIR, "scaler.pkl"), "rb") as f:
                            saved_scaler = pickle.load(f)
                        
                        # Load all model .pkl files, restoring original names
                        loaded_models = {}
                        for fname in saved_files:
                            if fname in ("scaler.pkl", "metadata.pkl"):
                                continue
                            # Reverse the safe_name transform: underscores back to spaces
                            original_name = fname.replace(".pkl", "").replace("_", " ")
                            with open(os.path.join(PRETRAINED_DIR, fname), "rb") as f:
                                loaded_models[original_name] = pickle.load(f)
                        
                        # Load metadata (results_df, roc_data)
                        meta_path = os.path.join(PRETRAINED_DIR, "metadata.pkl")
                        if os.path.isfile(meta_path):
                            with open(meta_path, "rb") as f:
                                meta = pickle.load(f)
                            results_df = meta["results_df"]
                            roc_data = meta["roc_data"]
                        else:
                            st.warning("Metadata file missing. Performance table may not be restored.")
                            results_df = pd.DataFrame()
                            roc_data = []
                        
                        # Recompute scaled test sets from loaded scaler
                        ldr_X_test_scaled = saved_scaler.transform(X_test)
                        ldr_X_train_scaled = saved_scaler.transform(X_train)
                        
                        # Store in session state
                        st.session_state.results_df = results_df
                        st.session_state.roc_data = roc_data
                        st.session_state.trained_models = loaded_models
                        st.session_state.ann = loaded_models.get("ANN")
                        st.session_state.scaler = saved_scaler
                        st.session_state.le_dict = le_dict
                        st.session_state.feature_names = X.columns.tolist()
                        st.session_state.numeric_cols = numeric_cols
                        st.session_state.categorical_cols = categorical_cols
                        st.session_state.df_encoded = df_encoded
                        st.session_state.X_train = X_train
                        st.session_state.X_train_scaled = ldr_X_train_scaled
                        st.session_state.X_test_scaled = ldr_X_test_scaled
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        
                        st.success(f"✅ Successfully loaded {len(loaded_models)} pretrained models!")
                    
                    except Exception as load_err:
                        st.error(f"Error loading pretrained models: {load_err}")
        
        else:
            # ---- TRAIN FRESH ----
            if st.button("🚀 Train All Models", key="train_button"):
                with st.spinner("Training models... This may take a moment..."):
                    # Train classical models
                    results_df, roc_data, trained_models = train_models(
                        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
                    )
                    
                    # Train ANN
                    ann, ann_metrics, ann_roc, history = train_ann(
                        X_train_scaled, X_test_scaled, y_train, y_test
                    )
                    
                    # Add ANN to results
                    results_df.loc[len(results_df)] = [
                        "ANN", ann_metrics[0], ann_metrics[1], ann_metrics[2],
                        ann_metrics[3], ann_metrics[4], ann_metrics[5],
                        ann_metrics[6], ann_metrics[7], ann_metrics[8]
                    ]
                    
                    roc_data.append(("ANN", ann_roc[0], ann_roc[1], ann_roc[2]))
                    trained_models["ANN"] = ann
                    
                    st.success("✅ Training completed!")
                    
                    # Store in session state
                    st.session_state.results_df = results_df
                    st.session_state.roc_data = roc_data
                    st.session_state.trained_models = trained_models
                    st.session_state.ann = ann
                    st.session_state.scaler = scaler
                    st.session_state.le_dict = le_dict
                    st.session_state.feature_names = X.columns.tolist()
                    st.session_state.numeric_cols = numeric_cols
                    st.session_state.categorical_cols = categorical_cols
                    st.session_state.df_encoded = df_encoded
                    st.session_state.X_train = X_train
                    st.session_state.X_train_scaled = X_train_scaled
                    st.session_state.X_test_scaled = X_test_scaled
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
        
        # Display results if available
        if 'results_df' in st.session_state:
            st.subheader("📈 Model Performance Results")
            
            results_sorted = st.session_state.results_df.sort_values(by="ROC-AUC", ascending=False)
            st.dataframe(results_sorted.style.highlight_max(axis=0), use_container_width=True)
            
            # Best model
            best_model_row = results_sorted.iloc[0]
            st.success(f"🏆 Best Model: **{best_model_row['Model']}** with ROC-AUC: {best_model_row['ROC-AUC']:.4f}")

            # Overfitting Gap Table
            st.subheader("📊 Overfitting Analysis (Train vs Test Accuracy Gap)")
            gap_df = results_sorted[["Model", "Train Accuracy", "Accuracy", "Gap"]].copy()
            gap_df = gap_df.rename(columns={"Accuracy": "Test Accuracy"})

            def gap_status(gap):
                if gap < 0.02:
                    return "✅ No overfitting"
                elif gap < 0.05:
                    return "✅ Mild — acceptable"
                elif gap < 0.10:
                    return "⚠️ Moderate — needs justification"
                else:
                    return "❌ Severe overfitting"

            gap_df["Status"] = gap_df["Gap"].apply(gap_status)
            gap_df["Train Accuracy"] = gap_df["Train Accuracy"].map("{:.6f}".format)
            gap_df["Test Accuracy"] = gap_df["Test Accuracy"].map("{:.6f}".format)
            gap_df["Gap"] = gap_df["Gap"].map("{:.6f}".format)
            st.dataframe(gap_df, use_container_width=True)
            
            # ---- Save Models to Disk ----
            st.markdown("---")
            st.subheader("💾 Save Trained Models")
            st.write("Save all trained models to disk so they can be loaded next time without retraining.")
            if st.button("💾 Save All Models to Disk", key="save_models_button"):
                try:
                    os.makedirs(PRETRAINED_DIR, exist_ok=True)
                    
                    # Save scaler
                    with open(os.path.join(PRETRAINED_DIR, "scaler.pkl"), "wb") as f:
                        pickle.dump(st.session_state.scaler, f)
                    
                    # Save each model separately
                    for name, model in st.session_state.trained_models.items():
                        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
                        with open(os.path.join(PRETRAINED_DIR, f"{safe_name}.pkl"), "wb") as f:
                            pickle.dump(model, f)
                    
                    # Save metadata (results_df & roc_data) — exclude non-picklable objects
                    meta = {
                        "results_df": st.session_state.results_df,
                        "roc_data": st.session_state.roc_data
                    }
                    with open(os.path.join(PRETRAINED_DIR, "metadata.pkl"), "wb") as f:
                        pickle.dump(meta, f)
                    
                    n_saved = len(st.session_state.trained_models)
                    st.success(f"✅ {n_saved} models saved to `{PRETRAINED_DIR}/` folder! You can now use 'Load Pretrained Models' next time.")
                except Exception as save_err:
                    st.error(f"Error saving models: {save_err}")
    
    except Exception as e:
        st.error(f"Error during training: {e}")



# ==========================================
# PAGE 3: MAKE PREDICTION
# ==========================================
elif page == "🔮 Make Prediction":
    st.subheader("🔮 Make Prediction")
    
    if 'scaler' not in st.session_state or 'trained_models' not in st.session_state:
        st.warning("⚠️ Please train the models first on the 'Model Training' page")
    else:
        try:
            # Get feature names and data
            feature_names = st.session_state.feature_names
            df_encoded = st.session_state.df_encoded
            numeric_cols = st.session_state.numeric_cols
            categorical_cols = [col for col in st.session_state.categorical_cols if col != df_encoded.columns[-1]]
            le_dict = st.session_state.le_dict
            
            st.write("Enter screening values for the patient:")
            
            # Create input form
            user_input = {}
            cols = st.columns(3)
            
            for idx, feature in enumerate(feature_names):
                with cols[idx % 3]:
                    # Get min/max from encoded data
                    min_val = float(df_encoded[feature].min())
                    max_val = float(df_encoded[feature].max())
                    default_val = (min_val + max_val) / 2
                    
                    user_input[feature] = st.slider(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=0.1
                    )
            
            if st.button("🎯 Predict", key="predict_button"):
                # Prepare input - convert to proper values
                input_values = []
                for f in feature_names:
                    val = user_input[f]
                    try:
                        input_values.append(float(val))
                    except:
                        input_values.append(val)
                
                input_array = np.array(input_values).reshape(1, -1).astype(float)
                input_scaled = st.session_state.scaler.transform(input_array)
                
                st.session_state.last_input_values = input_values
                st.session_state.last_input_array = input_array
                st.session_state.last_input_scaled = input_scaled
                st.session_state.has_prediction = True

            if st.session_state.get("has_prediction", False):
                input_values = st.session_state.last_input_values
                input_array = st.session_state.last_input_array
                input_scaled = st.session_state.last_input_scaled
                
                col1, col2 = st.columns(2)
                
                results_sorted = st.session_state.results_df.sort_values(by="ROC-AUC", ascending=False)
                best_classical = results_sorted[results_sorted['Model'] != 'ANN'].iloc[0]
                best_model = st.session_state.trained_models[best_classical['Model']]
                prediction = best_model.predict(input_scaled)[0]
                probability = best_model.predict_proba(input_scaled)[0][1]

                with col1:
                    st.subheader("Classical Model Predictions")
                    st.write(f"**Model:** {best_classical['Model']}")
                    st.write(f"**Prediction:** {'🔴 ASD Positive' if prediction == 1 else '🟢 ASD Negative'}")
                    st.write(f"**Confidence:** {probability*100:.2f}%")
                
                with col2:
                    st.subheader("ANN Model Prediction")
                    ann_prob = st.session_state.ann.predict_proba(input_scaled)[0][1]
                    ann_pred = 1 if ann_prob > 0.5 else 0
                    
                    st.write(f"**Model:** Artificial Neural Network")
                    st.write(f"**Prediction:** {'🔴 ASD Positive' if ann_pred == 1 else '🟢 ASD Negative'}")
                    st.write(f"**Confidence:** {ann_prob*100:.2f}%")
                
                st.markdown("---")
                st.subheader("💡 Explainable AI (XAI) & Clinical Decision Support")
                
                # Model selection dropdown for XAI explanation
                xai_model_name = st.selectbox(
                    "Select Model to Explain:",
                    options=list(st.session_state.trained_models.keys()),
                    index=0,
                    key="xai_model_select"
                )
                selected_xai_model = st.session_state.trained_models[xai_model_name]
                selected_prob = selected_xai_model.predict_proba(input_scaled)[0][1]
                
                # Render XAI Tabs
                xai_tab1, xai_tab2, xai_tab3, xai_tab4 = st.tabs([
                    "🩺 Clinical Narrative", 
                    "📊 SHAP Feature Attribution", 
                    "🍋 LIME Explanation", 
                    "🔄 What-If Simulator"
                ])
                
                with st.spinner("Computing Explainable AI (XAI) attributions..."):
                    shap_exp, shap_values, base_val = compute_shap_explanation(
                        selected_xai_model, input_scaled, feature_names,
                        st.session_state.X_train_scaled, input_values
                    )
                
                with xai_tab1:
                    if shap_values is not None and base_val is not None:
                        narrative = generate_clinical_narrative(
                            feature_names, input_values, shap_values,
                            base_val, selected_prob, xai_model_name
                        )
                        st.markdown(narrative)
                    else:
                        st.info("Clinical narrative unavailable due to SHAP computation limitation.")
                
                with xai_tab2:
                    st.write(f"### SHAP Feature Attribution Waterfall ({xai_model_name})")
                    if shap_exp is not None:
                        try:
                            fig_wf = plt.figure(figsize=(9, 5))
                            shap.plots.waterfall(shap_exp, max_display=10, show=False)
                            st.pyplot(plt.gcf())
                            plt.close('all')
                        except Exception as w_err:
                            st.warning(f"SHAP Waterfall rendering note: {w_err}")
                        
                        st.write("### Feature Contribution Bar Plot")
                        try:
                            fig_bar, ax_bar = plt.subplots(figsize=(9, 5))
                            sorted_indices = np.argsort(np.abs(shap_values))[::-1][:10]
                            top_features = [feature_names[i] for i in sorted_indices]
                            top_vals = [shap_values[i] for i in sorted_indices]
                            colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in top_vals]
                            ax_bar.barh(top_features[::-1], top_vals[::-1], color=colors[::-1])
                            ax_bar.set_xlabel("SHAP Value (Impact on Risk Probability)")
                            ax_bar.set_title(f"Top 10 Feature Contributions ({xai_model_name})")
                            plt.tight_layout()
                            st.pyplot(fig_bar)
                            plt.close('all')
                        except Exception as b_err:
                            st.warning(f"SHAP bar plot rendering note: {b_err}")
                    else:
                        st.warning("SHAP plot computation encountered an issue.")

                
                with xai_tab3:
                    st.write(f"### LIME (Local Interpretable Model-agnostic Explanations) ({xai_model_name})")
                    lime_exp = compute_lime_explanation(
                        selected_xai_model, input_scaled, feature_names, st.session_state.X_train_scaled
                    )
                    if lime_exp is not None:
                        fig_lime = lime_exp.as_pyplot_figure()
                        plt.tight_layout()
                        st.pyplot(fig_lime)
                    else:
                        st.warning("LIME explanation unavailable.")
                
                with xai_tab4:
                    st.write("### 🔄 Interactive What-If Sensitivity Simulator")
                    st.info("Modify screening scores dynamically to test how individual patient inputs affect predicted ASD risk confidence in real time.")
                    
                    target_feat = st.selectbox("Select Feature to Modify:", feature_names, key="whatif_feat")
                    feat_idx = feature_names.index(target_feat)
                    
                    current_val = float(input_values[feat_idx])
                    min_val = float(df_encoded[target_feat].min())
                    max_val = float(df_encoded[target_feat].max())
                    
                    simulated_val = st.slider(
                        f"Simulated Value for {target_feat}:",
                        min_value=min_val,
                        max_value=max_val,
                        value=current_val,
                        step=0.1,
                        key="whatif_slider"
                    )
                    
                    if simulated_val != current_val:
                        mod_input = list(input_values)
                        mod_input[feat_idx] = simulated_val
                        mod_array = np.array(mod_input).reshape(1, -1).astype(float)
                        mod_scaled = st.session_state.scaler.transform(mod_array)
                        
                        orig_prob = selected_xai_model.predict_proba(input_scaled)[0][1]
                        sim_prob = selected_xai_model.predict_proba(mod_scaled)[0][1]
                        diff = sim_prob - orig_prob
                        
                        mcol1, mcol2, mcol3 = st.columns(3)
                        mcol1.metric("Original Confidence", f"{orig_prob*100:.2f}%")
                        mcol2.metric("Simulated Confidence", f"{sim_prob*100:.2f}%")
                        mcol3.metric("Probability Shift", f"{diff*100:+.2f}%", delta_color="inverse")
        
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")


# ==========================================
# PAGE 4: MODEL COMPARISON
# ==========================================
elif page == "📊 Model Comparison":
    st.subheader("📊 Model Comparison & Visualization")
    
    if 'results_df' not in st.session_state:
        st.warning("⚠️ Please train the models first on the 'Model Training' page")
    else:
        try:
            results_df = st.session_state.results_df.sort_values(by="ROC-AUC", ascending=False)
            roc_data = st.session_state.roc_data
            
            # Tabs for different visualizations
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Performance Metrics", "ROC Curves", "Bar Chart", "Model Ranking", "💡 Global XAI & Importance"
            ])
            
            with tab1:
                st.subheader("Detailed Metrics Table")
                st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)
            
            with tab2:
                st.subheader("ROC Curve Comparison")
                fig, ax = plt.subplots(figsize=(10, 8))
                
                for name, fpr, tpr, auc in roc_data:
                    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", linewidth=2)
                
                ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
                ax.set_xlabel("False Positive Rate", fontsize=12)
                ax.set_ylabel("True Positive Rate", fontsize=12)
                ax.set_title("ROC Curve Comparison (All Models)", fontsize=14, fontweight='bold')
                ax.legend(loc='lower right')
                ax.grid(alpha=0.3)
                
                st.pyplot(fig)
            
            with tab3:
                st.subheader("Model Performance Comparison")
                fig, ax = plt.subplots(figsize=(12, 6))
                
                metrics_to_plot = ["Accuracy", "F1 Score", "ROC-AUC"]
                results_df.set_index("Model")[metrics_to_plot].plot(kind='bar', ax=ax)
                
                ax.set_title("Model Performance Comparison", fontsize=14, fontweight='bold')
                ax.set_ylabel("Score", fontsize=12)
                ax.set_xlabel("Model", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                st.pyplot(fig)
            
            with tab4:
                st.subheader("Model Ranking by ROC-AUC")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sorted_results = results_df.sort_values("ROC-AUC", ascending=True)
                colors = ['#2ecc71' if i == len(sorted_results) - 1 else '#3498db' 
                         for i in range(len(sorted_results))]
                
                ax.barh(sorted_results['Model'], sorted_results['ROC-AUC'], color=colors)
                ax.set_xlabel("ROC-AUC Score", fontsize=12)
                ax.set_title("Model Ranking by ROC-AUC", fontsize=14, fontweight='bold')
                ax.set_xlim([0, 1])
                
                for i, v in enumerate(sorted_results['ROC-AUC']):
                    ax.text(v + 0.02, i, f'{v:.4f}', va='center')
                
                st.pyplot(fig)
                
            with tab5:
                st.subheader("💡 Global Feature Importance & XAI Analysis")
                st.markdown("Analyze how features impact model predictions globally across the entire screening dataset.")
                
                global_model_name = st.selectbox(
                    "Select Model for Feature Importance Analysis:",
                    options=list(st.session_state.trained_models.keys()),
                    key="global_xai_model_select"
                )
                g_model = st.session_state.trained_models[global_model_name]
                
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    st.markdown("#### Permutation Feature Importance")
                    with st.spinner("Calculating permutation importance across test dataset..."):
                        try:
                            p_res = permutation_importance(
                                g_model,
                                st.session_state.X_test_scaled,
                                st.session_state.y_test,
                                n_repeats=5,
                                random_state=42
                            )
                            perm_sorted_idx = p_res.importances_mean.argsort()
                            
                            fig_perm, ax_perm = plt.subplots(figsize=(8, 6))
                            ax_perm.barh(
                                np.array(st.session_state.feature_names)[perm_sorted_idx],
                                p_res.importances_mean[perm_sorted_idx],
                                color='#3498db'
                            )
                            ax_perm.set_xlabel("Mean Accuracy Drop when Feature Shuffled")
                            ax_perm.set_title(f"Permutation Importance ({global_model_name})")
                            plt.tight_layout()
                            st.pyplot(fig_perm)
                        except Exception as pe:
                            st.warning(f"Could not compute Permutation Importance: {pe}")
                            
                with col_g2:
                    st.markdown("#### Feature Importance Summary")
                    if hasattr(g_model, "feature_importances_"):
                        g_importances = g_model.feature_importances_
                        g_df = pd.DataFrame({
                            'Feature': st.session_state.feature_names,
                            'Importance Score': g_importances
                        }).sort_values(by='Importance Score', ascending=False)
                        
                        fig_gimp, ax_gimp = plt.subplots(figsize=(8, 6))
                        ax_gimp.barh(g_df['Feature'][::-1], g_df['Importance Score'][::-1], color='#9b59b6')
                        ax_gimp.set_xlabel("Tree / Gini Importance Score")
                        ax_gimp.set_title(f"Built-in Feature Importance ({global_model_name})")
                        plt.tight_layout()
                        st.pyplot(fig_gimp)
                    else:
                        if 'p_res' in locals():
                            g_df = pd.DataFrame({
                                'Feature': st.session_state.feature_names,
                                'Importance Mean': p_res.importances_mean,
                                'Importance Std': p_res.importances_std
                            }).sort_values(by='Importance Mean', ascending=False)
                            st.dataframe(g_df, use_container_width=True)
                        else:
                            st.info("Feature ranking unavailable.")

        
        except Exception as e:
            st.error(f"Error during visualization: {e}")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='font-size: 12px; color: gray;'>
    Autism Spectrum Disorder Screening Prediction System | Machine Learning & ANN | Educational Purpose Only
    </p>
</div>
""", unsafe_allow_html=True)
