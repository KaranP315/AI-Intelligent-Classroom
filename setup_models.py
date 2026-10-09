"""Project setup helpers: device selection and model bootstrapping."""

import os

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

XGBClassifier._estimator_type = "classifier"


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def models_exist(base_dir="."):
    model_dir = os.path.join(base_dir, "trained_models")
    return (
        os.path.exists(os.path.join(model_dir, "bimodel.pt"))
        and os.path.exists(os.path.join(model_dir, "xgboost_classifier.json"))
    )


def ensure_trained_models(base_dir=".", force=False):
    """
    Ensure SSCon model artifacts exist.

    If data/*.pkl files are present, run the full SSCon training pipeline.
    Otherwise, bootstrap lightweight demo models so inference can run.
    """
    if models_exist(base_dir) and not force:
        print("[Setup] Trained models already present.")
        return

    os.makedirs(os.path.join(base_dir, "trained_models"), exist_ok=True)
    data_dir = os.path.join(base_dir, "data")
    required = [
        "labeled_training_data.pkl",
        "unlabeled_training_data.pkl",
        "validation_data.pkl",
        "test_data.pkl",
    ]

    if all(os.path.exists(os.path.join(data_dir, name)) for name in required):
        print("[Setup] Found training pickles  running SSCon training...")
        import runpy

        runpy.run_path(os.path.join(base_dir, "main.py"), run_name="__main__")
        return

    print(
        "[Setup] Training data not found in data/. "
        "Creating bootstrap demo models so the pipeline can start."
    )
    _create_bootstrap_models(base_dir)


def _create_bootstrap_models(base_dir):
    from train_biencoder import HiddenLayerBiEncoder

    rng = np.random.default_rng(42)
    n_per_class = 40
    feature_dim = 1408
    embed_dim = 384
    labels = np.array([0, 1, 2, 3])
    X = []
    y = []
    for label in labels:
        center = rng.normal(size=feature_dim)
        samples = center + rng.normal(scale=0.15, size=(n_per_class, feature_dim))
        X.append(samples.astype(np.float32))
        y.append(np.full(n_per_class, label, dtype=np.int32))
    X = np.vstack(X)
    y = np.concatenate(y)

    device = get_torch_device()
    model = HiddenLayerBiEncoder(feature_dim, embed_dim).to(device)
    model.eval()

    with torch.no_grad():
        projected = model(torch.tensor(X, device=device)).cpu().numpy()

    xgb = XGBClassifier(
        objective="multi:softprob",
        n_estimators=120,
        num_class=4,
        tree_method="hist",
    )
    xgb.fit(projected.astype(np.float32), y)

    model_dir = os.path.join(base_dir, "trained_models")
    torch.save(model, os.path.join(model_dir, "bimodel.pt"))
    xgb.save_model(os.path.join(model_dir, "xgboost_classifier.json"))
    print(f"[Setup] Bootstrap models saved to {model_dir}/")

    # Also write minimal demo pickles for test_gpu.py compatibility checks
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    demo_df = pd.DataFrame({"combined_embed": list(X[:10]), "act": y[:10]})
    unlabeled_df = pd.DataFrame({"combined_embed": list(X[10:20])})
    for name, df in [
        ("labeled_training_data.pkl", demo_df),
        ("unlabeled_training_data.pkl", unlabeled_df),
        ("validation_data.pkl", demo_df),
        ("test_data.pkl", demo_df),
    ]:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            df.to_pickle(path)


if __name__ == "__main__":
    ensure_trained_models(os.path.dirname(os.path.abspath(__file__)))
