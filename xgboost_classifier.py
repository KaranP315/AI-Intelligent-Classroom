# Get XGBoost Trained classifier
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score
)
from xgboost import XGBClassifier
XGBClassifier._estimator_type = "classifier"


"""
    Trains an XGBoost classifier model.

    Args:
        featureCol (str): Name of the feature column in the input dataframe
        labelCol (str): Name of the label column in the input dataframe
        inp_df (pd.DataFrame): Input dataframe for training the classifier
        val_df (pd.DataFrame): Validation dataframe for evaluation
        labelList (list): List of possible labels
        acceleration (str): Acceleration device to use, either "cpu" or "gpu" (default: "cpu")

    Returns:
        dict: Dictionary containing the trained XGBoost classifier, predicted labels,
              predicted label probabilities, and class label order
    """
def get_classifier_model(featureCol, labelCol, inp_df, val_df,
                         labelList,
                         acceleration = "cpu"
                         ):

  if acceleration == 'cuda':
    tree_method = 'gpu_hist'
  else:
    # MPS (Apple Silicon) and CPU both use 'hist' — XGBoost has no MPS backend
    tree_method = 'hist'
                          
  params = {
              'objective':'multi:softprob',
              'n_estimators':500,
              'num_class':len(labelList),
              "tree_method": tree_method
          }         
            
  # instantiate the classifier 
  xgb_clf = XGBClassifier(**params)
  
  X_train = np.ascontiguousarray(np.vstack(inp_df[featureCol].values).astype(np.float32))
  y_train = np.ascontiguousarray(np.array(inp_df[labelCol].values, dtype=np.int32))

  X_test = np.ascontiguousarray(np.vstack(val_df[featureCol].values).astype(np.float32))
  
  xgb_clf.fit(X_train, y_train)
  y_pred = xgb_clf.predict(X_test)
  y_pred_prob = xgb_clf.predict_proba(X_test)

  return {
      'xgb_classifier': xgb_clf,
      'y_pred': y_pred,
      'y_pred_prob': y_pred_prob,
      'prob_label_order': xgb_clf.classes_
  }  


  """
  Creates a distilled dataset based o n the predicted labels and probabilities.

  Args:
      inp_df (pd.DataFrame): Input dataframe to extract the original data
      y_pred (np.ndarray): Predicted labels for the input dataframe
      y_pred_prob (np.ndarray): Predicted label probabilities for the input dataframe
      prob_thrld (float): Probability threshold to filter the data (default: 0.85)

  Returns:
      pd.DataFrame: Distilled dataset with added columns for predicted labels, predicted label
                    probabilities, and maximum probability values
  """
def distilled_dataset(inp_df, y_pred, y_pred_prob, prob_thrld = 0.85):
  soft_label_df = inp_df.copy()
  soft_label_df['soft_label'] = y_pred
  soft_label_df['soft_label_prob'] = y_pred_prob
  soft_label_df['max_prob'] = soft_label_df.soft_label_prob.apply(lambda x: np.max(x))
  soft_label_df = soft_label_df[soft_label_df.max_prob>prob_thrld]
  return soft_label_df


  """
  Calculates and prints the performance metrics of a classifier model.

  Args:
      y_test (np.ndarray): Ground truth labels
      y_pred (np.ndarray): Predicted labels
      print_stats (bool): Whether to print the performance metrics (default: True)

  Returns:
      dict: Dictionary containing the calculated performance metrics
  """
def print_model_performance(y_test, y_pred, print_stats=True):
    res = {
        "mcc": matthews_corrcoef(y_test,y_pred),
        "F1_macro": f1_score(y_test,y_pred,
                        average='macro'),
        "F1_micro":f1_score(y_test,y_pred,
                        average='micro'),
        "Precision_macro": precision_score(y_test,y_pred,
                                        average='macro'),
        "Recall_macro": recall_score(y_test,y_pred,
                                            average='macro')
    }

    if print_stats:
        print(res)
    return res
