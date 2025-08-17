import torch 
import os
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_PATH = "mnist"
MODEL_PATH = os.path.join("mnist","best_model.pth")
NUM_EPOCHS = 20
BATCH_SIZE = 128 
NUM_WORKERS = 4 if torch.cuda.is_available() else 2
