import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import io
from openai import AzureOpenAI

# Initialize Azure OpenAI client
endpoint = os.getenv("ENDPOINT_URL", "URL")
deployment = os.getenv("DEPLOYMENT_NAME", "o4-mini")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "YOUR KEY")
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",
)

def generate_analysis(raw_df, preprocessed_df, stored_models, target_col=None):
    """Send dataset and models info to Azure OpenAI and get a summarized analysis."""

    # Capture dataset info
    buffer_raw = io.StringIO()
    raw_df.info(buf=buffer_raw)
    raw_info = buffer_raw.getvalue()

    buffer_pre = io.StringIO()
    preprocessed_df.info(buf=buffer_pre)
    pre_info = buffer_pre.getvalue()

    # Basic statistics
    raw_stats = f"""
Raw Data Shape: {raw_df.shape}
Preprocessed Data Shape: {preprocessed_df.shape}
Columns Removed: {len(raw_df.columns) - len(preprocessed_df.columns)}
Target Column: {target_col}
"""

    # Prepare models info
    models_summary = {}
    for name, info in stored_models.items():
        models_summary[name] = {
            "accuracy": info.get("accuracy", 0),
            "f1_weighted": info.get("f1_weighted", 0),
            "best_params": info.get("best_params", {}),
        }

    # Compose prompt text
    prompt_text = f"""
You are an expert data scientist. Analyze the following machine learning pipeline results and provide insights.

DATASET OVERVIEW:
{raw_stats}

RAW DATA INFO:
{raw_info}

PREPROCESSED DATA INFO:
{pre_info}

MODEL PERFORMANCE:
{json.dumps(models_summary, indent=2)}

Please provide a comprehensive analysis covering:

1. **Data Quality Assessment**:
   - Data completeness and potential issues
   - Impact of preprocessing steps
   - Feature engineering effectiveness

2. **Model Performance Analysis**:
   - Comparative performance of different models
   - Strengths and weaknesses of each algorithm
   - Parameter optimization insights

3. **Recommendations**:
   - Suggestions for improving model performance
   - Potential next steps (feature engineering, different algorithms, etc.)
   - Data collection recommendations if applicable

4. **Business Insights**:
   - Practical implications of the results
   - Model deployment considerations

Provide your analysis in a clear, structured format with bullet points where appropriate.
"""

    try:
        # Generate completion - REMOVED temperature parameter
        completion = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt_text}],
            max_completion_tokens=2000
            # Removed temperature parameter as it's not supported by this model
        )

        # Extract analysis text
        analysis_text = completion.choices[0].message.content
        return analysis_text

    except Exception as e:
        return f"Error generating AI analysis: {str(e)}\n\nPlease check your Azure OpenAI configuration."

def generate_roc_analysis(stored_models, y_true, model_names=None):
    """Generate ROC curve analysis for stored models"""
    try:
        if not stored_models:
            return None, "No models available for ROC analysis"
        
        y_true = np.array(y_true)
        classes = np.unique(y_true)
        n_classes = len(classes)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        roc_data = {}
        
        for model_name, info in stored_models.items():
            if model_names and model_name not in model_names:
                continue
                
            y_test = np.array(info.get('y_test', []))
            y_pred = np.array(info.get('y_pred', []))
            X_test = info.get('X_test')
            
            if len(y_test) == 0 or X_test is None:
                continue
            
            # For ROC, we need probability scores
            # Since we don't store the actual model objects, we'll use the predictions
            # This is a simplified version - in production, you'd want to store probability scores
            
            if n_classes == 2:
                # For binary classification, use predictions as scores
                y_score = y_pred
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
                roc_data[model_name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc}
        
        if not roc_data:
            return None, "No valid models for ROC analysis"
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison')
        ax.legend(loc="lower right")
        ax.grid(True)
        
        return fig, roc_data
        
    except Exception as e:
        return None, f"Error generating ROC analysis: {str(e)}"
        