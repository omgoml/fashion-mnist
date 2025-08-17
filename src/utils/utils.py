import torch 
import torch.nn as nn  
import torch.optim as optim 
import os 
from config.env import *
from model.cnn import MNISTModel


def save_model(epoch:int, model:nn.Module, optimizer: optim.Optimizer, best_accuracy, model_path):
    torch.save({
        "epoch": epoch, 
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_accuracy": best_accuracy
    },model_path)

    print(f"Model saved successfully to path {model_path}")

def load_model():

    if not os.path.exists(MODEL_PATH):
        print(f"Model path {MODEL_PATH} not exits")
        return None, None 

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = MNISTModel().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Model loaded from {MODEL_PATH}")
    print(f"Best tested accuracy: {checkpoint["best_accuracy"]}")
    print(f"Trained for {checkpoint["epoch"]} epochs")

    return model, checkpoint

