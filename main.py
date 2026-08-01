import os
os.environ["OMP_NUM_THREADS"] = "1"
from SSCon import SSCon_run
import torch
import pandas as pd
from xgboost import XGBClassifier
XGBClassifier._estimator_type = "classifier"
from xgboost_classifier import print_model_performance
import numpy as np
import os


def main():
    # Your data preprocessing code goes here...

    # Define the input arguments for SSCon_run
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    unlabeled_train_df = pd.read_pickle("data/unlabeled_training_data.pkl")  # DataFrame of unlabeled training data
    labeled_train_df = pd.read_pickle("data/labeled_training_data.pkl")  # DataFrame of labeled training data
    val_df = pd.read_pickle("data/validation_data.pkl")  # DataFrame of validation data
    test_df = pd.read_pickle("data/test_data.pkl") # Dataframe for test data
    featureCol = "combined_embed"  # Name of the feature column in the DataFrames
    labelCol = 'act'
    labelList = labeled_train_df[labelCol].unique().tolist()

    # --- Print hyperparameter configuration ---
    print("=" * 60)
    print("SSCon Training Pipeline — Hyperparameter Configuration")
    print("=" * 60)
    print(f"  Device:              {device}")
    print(f"  PyTorch version:     {torch.__version__}")
    print(f"  Feature column:      {featureCol}")
    print(f"  Label column:        {labelCol}")
    print(f"  Num classes:         {len(labelList)}")
    print(f"  Label list:          {labelList}")
    print(f"  HL_FEATURE_DIM:      1408")
    print(f"  EMBED_FEATURE_DIM:   384")
    print(f"  prob_thrld:          85")
    print(f"  biencoder_epoch:     5")
    print(f"  iteration_cnt:       5")
    print(f"  biencoder_path:      trained_models")
    print(f"  Labeled train size:  {len(labeled_train_df)}")
    print(f"  Unlabeled train sz:  {len(unlabeled_train_df)}")
    print(f"  Validation size:     {len(val_df)}")
    print(f"  Test size:           {len(test_df)}")
    print("=" * 60)

    # Ensure output directory exists
    os.makedirs("trained_models", exist_ok=True)

    results = SSCon_run(
    device,
    unlabeled_train_df,
    labeled_train_df,
    val_df,
    featureCol = featureCol,
    labelList = labelList,
    labelCol = labelCol,
    HL_FEATURE_DIM = 1408,
    prob_thrld = 85,
    biencoder_path = "trained_models",
    biencoder_epoch = 5,
    iteration_cnt = 5

    )




    # Print Test Data Results 
    params = {
                'objective':'multi:softprob',
                'n_estimators':500,
                'num_class':len(labelList),
                "tree_method": 'hist'
            }         
                
    # instantiate the classifier 
    xgb_clf = XGBClassifier(**params)
    xgb_clf.load_model("trained_models/xgboost_classifier.json")

    bimodel = torch.load("trained_models/bimodel.pt", weights_only=False).to(device)
    bimodel.eval()

    tst = torch.tensor(np.ascontiguousarray(np.vstack(test_df[featureCol].values).astype(np.float32))).to(device)
    X_test = np.ascontiguousarray(np.vstack(list(bimodel(tst).cpu().detach().numpy())).astype(np.float32))

    print("Final Pipeline Run Test Stats:")
    import xgboost as xgb
    preds_prob = xgb_clf.get_booster().predict(xgb.DMatrix(X_test))
    preds = np.argmax(preds_prob, axis=1)
    res = print_model_performance(test_df[labelCol].values, preds)



if __name__ == "__main__":
    main()
