from train_biencoder import train_hidden_layer_biencoder
from xgboost_classifier import (
    get_classifier_model,
    distilled_dataset,
    print_model_performance
)

from datetime import datetime
import torch
import pandas as pd
import numpy as np


def SSCon_run(
  device,                   # Device to run the function on (e.g., 'cpu' or 'mps')
  unlabeled_train_df,       # DataFrame of unlabeled training data
  labeled_train_df,         # DataFrame of labeled training data
  val_df,                   # DataFrame of validation data
  featureCol,               # Name of the feature column in the DataFrames
  labelList,                # List of possible labels
  labelCol,                 # Name of the label column in the DataFrames
  embedCol = "gen_hl_embed", # Name of the embedding column in the DataFrames
  HL_FEATURE_DIM = 1408,       # Dimension of the Feature Input Collumn
  EMBED_FEATURE_DIM = 384,     # Dimension of the encoder embedding features
  prob_thrld = 85,             # Probability threshold for distillation of high probability labels
  biencoder_path = "trained_models/biencoder",   # Path to saving best model
  biencoder_epoch = 5,                           # Number of epochs for training the biencoder model
  iteration_cnt = 5                              # Number of iterations for training the SSCon model
):
  """
  Function to run the SSCon model for semi-supervised learning.
  Returns:
    The MCC of the best model 
    Perfomance stats of the iterations
    Path to where the best models are saved
  """  
  date_time = datetime.now().strftime("%Y:%m:%d:%H:%M:%S")
  best_mcc = 0
  best_model = None
  bi_model = None
  perf_results = []
  accelerator = device.type


  unlabeled_df = unlabeled_train_df.copy()
  distill_df = labeled_train_df.copy()
  new_val_df = val_df.copy()

  clf_result = get_classifier_model(featureCol, labelCol,
                                    labeled_train_df,
                                    new_val_df,
                                    labelList,
                                    acceleration = accelerator)
  print("test_stats")
  res = print_model_performance(new_val_df[labelCol].values, clf_result['y_pred'])
  res['datetime'] = date_time
  res['iteration'] = 0
  perf_results.append(res)
  print("*******")
  for i in range(iteration_cnt):
    print(f"running iteration {i}")
    # Stage 1 context classifier
    clf_result = get_classifier_model(featureCol, labelCol, distill_df,
                                      unlabeled_df, labelList,
                                       acceleration = accelerator)

    #Distillation of high confindence labels
    distill_df = distilled_dataset(unlabeled_df, clf_result['y_pred'],
                                list(clf_result['y_pred_prob']),
                                prob_thrld = prob_thrld/100.0)
    distill_df[labelCol] = distill_df['soft_label']
    distill_df = pd.concat([distill_df, labeled_train_df], ignore_index=True)
    

    # Stage 2 contrastive Encoder
    bi_model = train_hidden_layer_biencoder(distill_df, labelList, featureCol,labelCol,
                                            max_epochs = biencoder_epoch,
                                            pretrained_model = bi_model,
                                            hl_features = HL_FEATURE_DIM,
                                            embed_features = EMBED_FEATURE_DIM,
                                            accelerator = accelerator
                                            ).to(device)
    
    # Validation of SSCon
    inp = torch.tensor(np.vstack(distill_df[featureCol].values)).to(device)
    new_x_train = bi_model(inp).cpu().detach().numpy()
    distill_df[embedCol] = list(new_x_train)

    inp = torch.tensor(np.vstack(unlabeled_df[featureCol].values)).to(device)
    new_x_train = bi_model(inp).cpu().detach().numpy()
    unlabeled_df[embedCol] = list(new_x_train)

    tst = torch.tensor(np.vstack(new_val_df[featureCol].values)).to(device)
    new_val_df[embedCol] = list(bi_model(tst).cpu().detach().numpy())

    clf_result = get_classifier_model(embedCol, labelCol,
                                  distill_df, new_val_df,
                                   labelList,
                                  acceleration = accelerator)

    print("test_stats")
    res = print_model_performance(new_val_df[labelCol].values,
                                  clf_result['y_pred'])
    
    if res['mcc'] > best_mcc:
      best_model = clf_result['xgb_classifier']
      best_mcc = res['mcc']
      torch.save(bi_model, f"{biencoder_path}/bimodel.pt")
      best_model.save_model(f"{biencoder_path}/xgboost_classifier.json")
  

    res['iteration'] = i+1
    perf_results.append(res)

    # Stage 2 Embedding Classifer
    clf_result = get_classifier_model(embedCol, labelCol,
                                  distill_df, unlabeled_df,
                                  labelList,  acceleration = accelerator)
    
    #Distillation of high confindence labels
    distill_df = distilled_dataset(unlabeled_df, clf_result['y_pred'],
                                list(clf_result['y_pred_prob']),
                                        prob_thrld = prob_thrld/100.0)
    distill_df[labelCol] = distill_df['soft_label']
    distill_df = pd.concat([distill_df, labeled_train_df], ignore_index=True)
    #TODO: update distill_df with given labels. 
  
  return {"best_mcc": best_mcc,
          "performance_stats": perf_results,
          "biencoder_model_path": biencoder_path}




