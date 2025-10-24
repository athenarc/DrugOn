from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, log_loss,
    balanced_accuracy_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import norm

from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

def plot_pdp_all_features(model_pipeline, X, feature_names=None):
    """
    Plots Partial Dependence Plots (PDPs) for all features using a fitted pipeline.

    Parameters:
    - model_pipeline: Fitted sklearn pipeline with a classifier at the end
    - X: Feature matrix used for fitting
    - feature_names: Optional list of feature names (default: X.columns if X is a DataFrame)
    """

    # Use provided feature names or infer from DataFrame
    if feature_names is None:
        try:
            features = list(X.columns)
        except AttributeError:
            features = [f"feature_{i}" for i in range(X.shape[1])]
    else:
        features = feature_names

    # Set figure size dynamically
    fig_height = len(features) * 2.5
    fig, ax = plt.subplots(
        nrows=len(features),
        ncols=1,
        figsize=(12, fig_height),
        constrained_layout=False
    )

    # Ensure ax is iterable
    if len(features) == 1:
        ax = [ax]

    # Plot PDPs
    display = PartialDependenceDisplay.from_estimator(
        model_pipeline,
        X,
        features=features,
        kind='average',
        grid_resolution=50,
        ax=ax
    )

    # Improve layout
    plt.tight_layout(pad=3.0, h_pad=2.0)
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.show()

def get_sorted_feature_importance(pipeline, feature_names):
    """
    Extracts and returns sorted feature importances from a fitted pipeline 
    with an XGBoost classifier as the final step.

    Parameters:
    - pipeline: Fitted sklearn pipeline (must have a 'xgb' step)
    - feature_names: List or index of feature names (e.g., X_train.columns)

    Returns:
    - importance_df: DataFrame sorted by importance (descending)
    """
    # Access the trained XGBoost model inside the pipeline
    xgb_model = pipeline.named_steps['xgb']

    # Extract raw importance values
    importances = xgb_model.feature_importances_

    # Build and sort the importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    return importance_df



def evaluate_classification_metrics(y_all,y_true, y_pred, y_prob, analysis_id=1, ratio=1, model_name="XGBoost", split_desc='80/20'):
    """
    Evaluates classification metrics and returns results in a dictionary.
    
    Parameters:
    - y_true: Ground truth labels
    - y_pred: Predicted labels
    - y_prob: Predicted probabilities for the positive class
    - analysis_id: Custom identifier for tracking (default=1)
    - ratio: Custom ratio value (optional)
    - model_name: Name of the model (default='XGBoost')
    - split_desc: Description of the train/test split
    
    Returns:
    - Dictionary of metrics
    """

    # Count positives and negatives
    negative = sum(y_all == 0)
    positive = sum(y_all == 1)

    # Confusion matrix handling
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = cm[0, 0] if y_true.iloc[0] == 0 else 0
        fp = cm[0, 1] if y_true.iloc[0] == 0 else 0
        fn = cm[1, 0] if y_true.iloc[0] == 1 else 0
        tp = cm[1, 1] if y_true.iloc[0] == 1 else 0

    # Type 1 and 2 errors
    type1 = fp / (fp + tn) if (fp + tn) > 0 else 0
    type2 = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Handle AUC and PR AUC safely
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = np.nan

    # Collect metrics
    result = {
        'AnalysisID': analysis_id,
        'Ratio': ratio,
        'Model': model_name,
        'Train-Test Split': split_desc,
        '# Positive Controls': positive,
        '# Negative Controls': negative,
        'AUC': auc,
        'PR AUC': pr_auc,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall (1-Type 2 error)': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'F1 Score (Macro)': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Type 1 Error (FPR)': type1,
        'Type 2 Error (FNR)': type2
    }
    return result