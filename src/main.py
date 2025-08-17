import torch 
import torch.optim as optim
import torch.nn as nn 
import numpy as np
from config.env import *
from data.DataHandler import data_handler
from inference.InferenceModel import InferenceModel
from model.cnn import MNISTModel
from train.TrainingProgress import TrainingModel 

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    IModel = InferenceModel()
        
    try:
        if IModel._load_trained_model() == "fail":

            model = MNISTModel().to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(),lr=0.001, weight_decay=1e-4)

            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=0.01,
                epochs=NUM_EPOCHS,
                steps_per_epoch=len(data_handler()[0]),
            )

            TrainModel = TrainingModel(model=model, scheduler=scheduler,optimizer=optimizer,criterion=criterion)
        
            TrainModel.train()
        
            print("\n loading trained model...")

            IModel._load_trained_model()
        else:
            while True:
                image_path = input("Enter a image path (or 'quit' to exist):")
                if image_path.lower() == "quit":
                    break
            
                try:
                    prediction, confidence, _ = IModel.predict_image_path(image_path)

                    print(f"Prediction: {prediction}")
                    print(f"Confidence: {confidence}")
                except Exception as e:
                    print(f"Error: {e}")
                    break

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


