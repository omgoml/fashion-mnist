import os
import torch.nn as nn 
from torchvision import transforms

from config.env import *
from utils.utils import load_model 

class InferenceModel:
    def __init__(self) -> None:
        self.model = None 

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ])
        
        self._load_trained_model()

    def _load_trained_model(self):
        
        if not os.path.exists(MODEL_PATH):
            print(f"Model path {MODEL_PATH} not exit")
            return "fail"

        try:
            loaded_model, checkpoint = load_model()

            if load_model is not None:
                self.model = loaded_model
                self.model.eval()

                print("Model loaded successfully")
                return "success"
            else:
                print("there are no trained model")
                print("please retrain")
                return "fail"
        
        except Exception as e:
            print(f"Error: {e}")
            return "fail"

    def predict_single_image(self, image_tensor: torch.Tensor):
        
        if self.model is None:
            print(f"Model not loaded")
            return None, None, None 

        if len(image_tensor.size()) == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = self.transform(image_tensor)

        with torch.no_grad():
            output = self.model(image_tensor)
            probability = torch.softmax(output,1)
            prediction = output.argmax(dim=1).item()
            confidence = probability.max().item()

        return prediction, confidence, probability

    def predict_image_path(self, image_path:str):

        if not os.path.exists(image_path):
            print(f"image path {image_path} not exist")
            return None, None, None
        
        from PIL import Image

        img = Image.open(image_path).convert("L")

        img = img.resize((28,28))

        image_tensor = torch.Tensor(self.transform(img))

        return self.predict_single_image(image_tensor)
