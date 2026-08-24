import os
import sys

# Prevent OpenMP segmentation faults
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from xgboost import XGBClassifier

# Monkeypatch for scikit-learn wrapper compatibility
XGBClassifier._estimator_type = "classifier"

# Import model architecture to ensure pickle load works
from train_biencoder import HiddenLayerBiEncoder


from setup_models import get_torch_device


def main():
    print("=" * 60)
    print("SSCon Dialogue Act Classifier Inference")
    print("=" * 60)

    # 1. Select Device
    device = get_torch_device()
    print(f"Using device: {device}")

    # 2. Load NLP models for embedding generation
    print("\nLoading Sentence-BERT model (all-MiniLM-L6-v2)...")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading DialoGPT-medium model and tokenizer (this may take a moment)...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    dialog_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/DialoGPT-medium"
    ).to(device)
    dialog_model.eval()

    # 3. Load Trained SSCon Models
    print("\nLoading trained SSCon bi-encoder and classifier...")
    if not os.path.exists("trained_models/bimodel.pt"):
        print("Error: trained_models/bimodel.pt not found.")
        sys.exit(1)
    if not os.path.exists("trained_models/xgboost_classifier.json"):
        print("Error: trained_models/xgboost_classifier.json not found.")
        sys.exit(1)

    bimodel = torch.load("trained_models/bimodel.pt", weights_only=False).to(
        device
    )
    bimodel.eval()

    xgb_clf = XGBClassifier()
    xgb_clf.load_model("trained_models/xgboost_classifier.json")

    # Mapping of predicted classes to Student Dialogue Acts (from the paper)
    act_mapping = {
        0: "Relating to another student",
        1: "Asking for more information",
        2: "Making a claim",
        3: "Providing evidence",
    }

    print("\nInitialization complete.")
    print("=" * 60)
    print("Dialogue session started!")
    print("Type your messages below.")
    print("Special Commands:")
    print("  /clear - Clear dialogue history/context")
    print("  /exit  - Exit the program")
    print("=" * 60)

    history = []

    while True:
        try:
            user_input = input("\nEnter Student Utterance: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            print("Exiting dialogue session. Goodbye!")
            break

        if user_input.lower() == "/clear":
            history = []
            print("Dialogue history cleared.")
            continue

        # Add current turn to history
        history.append(user_input)

        # We keep up to 10 utterances (current + 9 preceding context turns)
        context_turns = history[-10:]

        # --- Generate DialoGPT Contextual Embedding (1024-dim) ---
        # Concatenate using EOS token as separator
        context_text = "".join(
            [turn + tokenizer.eos_token for turn in context_turns]
        )

        inputs = tokenizer(context_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dialog_model(**inputs, output_hidden_states=True)

        # Extract hidden state of the EOS token from layer 12 (matching training distribution)
        hi = outputs.hidden_states[12][0, -1, :].cpu().numpy()

        # --- Generate Sentence-BERT Embedding (384-dim) ---
        si = sbert_model.encode(user_input)

        # --- Concatenate Embeddings (1408-dim) ---
        combined_embed = np.concatenate([hi, si], axis=0).astype(np.float32)

        # --- Pass through Bi-Encoder to project to 384-dim space ---
        x_in = torch.tensor(combined_embed).unsqueeze(0).to(device)
        with torch.no_grad():
            x_proj = bimodel(x_in).cpu().numpy().astype(np.float32)

        # Ensure contiguous layout for XGBoost
        x_proj_contiguous = np.ascontiguousarray(x_proj)

        # --- Predict dialogue act using XGBoost classifier ---
        import xgboost as xgb

        dmat = xgb.DMatrix(x_proj_contiguous)
        preds_prob = xgb_clf.get_booster().predict(dmat)[0]
        pred_class = np.argmax(preds_prob)

        # --- Display Results ---
        print("\n--- SSCon Prediction ---")
        print(f"Predicted Act Index: {pred_class}")
        print(
            f"Predicted Act Label: {act_mapping.get(pred_class, 'Unknown Act')}"
        )
        print("Probability Distribution:")
        for idx, prob in enumerate(preds_prob):
            label_desc = act_mapping.get(idx, "Unknown")
            print(f"  Class {idx} ({label_desc}): {prob*100:.2f}%")
        print("-" * 25)


if __name__ == "__main__":
    main()
