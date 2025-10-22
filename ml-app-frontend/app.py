import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import io
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

BACKEND_URL = "http://127.0.0.1:8000"  # Change if hosted elsewhere

st.set_page_config(layout="wide")
st.title("CSV Loader, Preprocessing & Model Evaluation (Backend)")

# -----------------------------
# SESSION STATE
# -----------------------------
for key in ['df', 'cleaned_df', 'stored_models', 'ai_analysis']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'stored_models' else {}

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
section = st.sidebar.radio(
    "Navigation",
    ["Upload Data", "Preprocess Data", "Run Model", "View Results", "AI Analysis"]
)

# -----------------------------
# UPLOAD DATA
# -----------------------------
if section == "Upload Data":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        # Reset the buffer position to start
        uploaded_file.seek(0)

        # Send file to backend
        files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
        response = requests.post(f"{BACKEND_URL}/data/upload", files=files)

        if response.ok:
            st.success("File uploaded successfully!")

            # Reset buffer again before reading with pandas
            uploaded_file.seek(0)
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.error(f"Upload failed: {response.text}")

    if st.session_state.df is not None:
        df = st.session_state.df
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.subheader("Raw Data Info")
        st.text(buffer.getvalue())
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

# -----------------------------
# PREPROCESS DATA
# -----------------------------
elif section == "Preprocess Data":
    if st.session_state.df is None:
        st.warning("Please upload a CSV first!")
    else:
        df = st.session_state.df
        target_col = st.selectbox("Select Target Column", df.columns)

        if st.button("Preprocess Data"):
            # Send as form data, not JSON
            payload = {"target_col": target_col}
            response = requests.post(f"{BACKEND_URL}/data/preprocess", data=payload)

            if response.ok:
                # Read CSV from response if your backend sends it as CSV
                cleaned_csv = response.content.decode("utf-8")
                st.session_state.cleaned_df = pd.read_csv(io.StringIO(cleaned_csv))
                st.session_state.target_col = target_col  # <-- store target column
                st.success("Data preprocessed successfully!")
            else:
                st.error(f"Preprocessing failed: {response.text}")

        if st.session_state.cleaned_df is not None:
            buffer2 = io.StringIO()
            st.session_state.cleaned_df.info(buf=buffer2)
            st.subheader("Preprocessed Data Info")
            st.text(buffer2.getvalue())
            st.subheader("Preprocessed Data Preview")
            st.dataframe(st.session_state.cleaned_df.head())


# -----------------------------
# RUN MODEL
# -----------------------------
elif section == "Run Model":
    if st.session_state.cleaned_df is None:
        st.warning("Preprocess data first!")
    else:
        model_choice = st.selectbox(
            "Select a model to run",
            ["Logistic Regression","KNN","SVM","Random Forest","XGBoost","Naive Bayes"]
        )

        if st.button("Train Selected Model"):
            # Convert cleaned_df to list of rows
            data_list = st.session_state.cleaned_df.values.tolist()
            columns_list = st.session_state.cleaned_df.columns.tolist()
            payload = {
                "model_name": model_choice,
                "data": data_list,
                "columns": columns_list,
                "target": st.session_state.target_col  # make sure target_col is stored in session
            }

            response = requests.post(f"{BACKEND_URL}/model/train", json=payload)
            if response.ok:
                res = response.json()["result"]
                st.session_state.stored_models[model_choice] = res
                st.success(f"{model_choice} trained successfully!")
            else:
                st.error(f"Training failed: {response.text}")


# -----------------------------
# VIEW RESULTS
# -----------------------------
elif section == "View Results":
    if not st.session_state.stored_models:
        st.warning("No models trained yet!")
    else:
        model_to_view = st.selectbox(
            "Select a model to view",
            list(st.session_state.stored_models.keys())
        )
        if model_to_view:
            info = st.session_state.stored_models[model_to_view]

            st.markdown("### Model Information")
            st.write(f"**Best Parameters:** {info['best_params']}")
            st.write(f"**Accuracy:** `{info['accuracy']:.4f}`")

            # Metrics
            macro = info["classification_report"]["macro avg"]
            weighted = info["classification_report"]["weighted avg"]
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Precision (Macro)", f"{macro['precision']:.3f}")
            with col2: st.metric("Recall (Macro)", f"{macro['recall']:.3f}")
            with col3: st.metric("F1-Score (Macro)", f"{macro['f1-score']:.3f}")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Precision (Weighted)", f"{weighted['precision']:.3f}")
            with col2: st.metric("Recall (Weighted)", f"{weighted['recall']:.3f}")
            with col3: st.metric("F1-Score (Weighted)", f"{weighted['f1-score']:.3f}")

            # Confusion matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(info["confusion_matrix"], annot=True, fmt="d", cmap="Greys", ax=ax)
            st.pyplot(fig)

# -----------------------------
# AI ANALYSIS
# -----------------------------
elif section == "AI Analysis":
    if not st.session_state.stored_models or st.session_state.df is None or st.session_state.cleaned_df is None:
        st.warning("Upload data, preprocess, and run at least one model first!")
    else:
        # Generate AI analysis
        if st.button("Generate AI Analysis"):
            with st.spinner("Generating AI analysis..."):
                try:
                    # Prepare data for backend
                    payload = {
                        "raw_data": st.session_state.df.values.tolist(),
                        "raw_columns": st.session_state.df.columns.tolist(),
                        "preprocessed_data": st.session_state.cleaned_df.values.tolist(),
                        "preprocessed_columns": st.session_state.cleaned_df.columns.tolist(),
                        "stored_models": st.session_state.stored_models,
                        "target_col": st.session_state.get('target_col', '')
                    }
                    
                    # Send to backend
                    response = requests.post(f"{BACKEND_URL}/analysis/anal", json=payload)
                    
                    if response.ok:
                        result = response.json()
                        st.session_state.ai_analysis = result["analysis"]
                        st.success("AI analysis generated!")
                    else:
                        st.error(f"AI analysis failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error generating analysis: {str(e)}")

        # Display AI analysis
        if st.session_state.ai_analysis:
            st.subheader("AI Analysis Report")
            st.markdown("### Comprehensive Model Analysis")
            st.text_area("Analysis Details", st.session_state.ai_analysis, height=400)
            
            # Add download button for the analysis
            analysis_text = st.session_state.ai_analysis
            st.download_button(
                label="Download Analysis Report",
                data=analysis_text,
                file_name="ml_analysis_report.txt",
                mime="text/plain"
            )

        # -----------------------------
        # Model Performance Comparison Chart
        # -----------------------------
        if st.session_state.stored_models:
            st.subheader("Model Performance Comparison")
            
            # Create performance comparison
            models = list(st.session_state.stored_models.keys())
            accuracies = [st.session_state.stored_models[model]['accuracy'] for model in models]
            f1_scores = [st.session_state.stored_models[model]['f1_weighted'] for model in models]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Accuracy comparison
            bars1 = ax1.bar(models, accuracies, color='skyblue', alpha=0.7)
            ax1.set_title('Model Accuracy Comparison')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
            
            # F1-score comparison
            bars2 = ax2.bar(models, f1_scores, color='lightcoral', alpha=0.7)
            ax2.set_title('Model F1-Score Comparison')
            ax2.set_ylabel('F1-Score (Weighted)')
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)

        # -----------------------------
        # ROC Curve Comparison
        # -----------------------------
        if st.session_state.stored_models:
            st.subheader("ROC Curve Comparison")
            
            # Check if we have probability scores for ROC curves
            models_with_proba = []
            for model_name, info in st.session_state.stored_models.items():
                if info.get('y_proba') is not None:
                    models_with_proba.append(model_name)
            
            if models_with_proba:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                for model_name in models_with_proba:
                    info = st.session_state.stored_models[model_name]
                    y_test = np.array(info['y_test'])
                    y_proba = np.array(info['y_proba'])
                    
                    # Handle binary and multi-class classification
                    n_classes = len(np.unique(y_test))
                    
                    if n_classes == 2:
                        # Binary classification - use positive class probabilities
                        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                            y_score = y_proba[:, 1]  # Positive class probabilities
                        else:
                            y_score = y_proba
                        
                        fpr, tpr, _ = roc_curve(y_test, y_score)
                        roc_auc = auc(fpr, tpr)
                        
                        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
                    
                    else:
                        # Multi-class classification - compute micro-average ROC
                        fpr = dict()
                        tpr = dict()
                        roc_auc = dict()
                        
                        # One-vs-Rest ROC curves
                        for i in range(n_classes):
                            fpr[i], tpr[i], _ = roc_curve(y_test == i, y_proba[:, i])
                            roc_auc[i] = auc(fpr[i], tpr[i])
                        
                        # Compute micro-average ROC curve and ROC area
                        fpr["micro"], tpr["micro"], _ = roc_curve(
                            label_binarize(y_test, classes=np.unique(y_test)).ravel(),
                            y_proba.ravel()
                        )
                        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                        
                        ax.plot(fpr["micro"], tpr["micro"],
                               label=f'{model_name} (micro-average AUC = {roc_auc["micro"]:.3f})')
                
                # Plot diagonal line
                ax.plot([0, 1], [0, 1], 'k--', lw=2)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve Comparison')
                ax.legend(loc="lower right")
                ax.grid(True)
                
                st.pyplot(fig)
            else:
                st.info("ROC curves are available for models that support probability predictions. Try training models like Logistic Regression, Random Forest, or SVM with probability=True.")

        # -----------------------------
        # Confusion Matrices
        # -----------------------------
        if st.session_state.stored_models:
            st.subheader("Confusion Matrices")
            
            # Let user select which model's confusion matrix to view
            model_to_view = st.selectbox(
                "Select model to view confusion matrix:",
                list(st.session_state.stored_models.keys())
            )
            
            if model_to_view:
                info = st.session_state.stored_models[model_to_view]
                cm = np.array(info['confusion_matrix'])
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title(f'Confusion Matrix - {model_to_view}')
                st.pyplot(fig)

        # -----------------------------
        # Feature Importance (if available)
        # -----------------------------
        if st.session_state.stored_models and 'Random Forest' in st.session_state.stored_models:
            st.subheader("Feature Importance (Random Forest)")
            
            # Note: In a full implementation, you would retrieve feature importance
            # from the trained model. This is a placeholder.
            st.info("Feature importance visualization would be displayed here when available.")