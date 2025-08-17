import torch.optim as optim
import torch.nn as nn 
from config.env import *
from data.DataHandler import data_handler 
from tqdm import tqdm
from utils.utils import save_model

class TrainingModel:
    def __init__(self, 
        model:nn.Module, 
        optimizer: optim.Optimizer,
        scheduler:optim.lr_scheduler.OneCycleLR,
        criterion,
        epochs: int = NUM_EPOCHS,
    ) -> None:
        self.train_loader, self.test_loader = data_handler()
        self.model = model 
        self.optimizer = optimizer 
        self.scheduler = scheduler 
        self.criterion = criterion
        self.epochs = epochs

        self.history = {
            "train_losses": [],
            "train_accuracies": [],
            "test_losses": [],
            "test_accuracies": [],
        } 

    def train_epoch(self):
        self.model.train() 
        running_loss = 0.0
        correct = 0 
        total = 0 

        pbar = tqdm(self.train_loader, desc="Training")

        for images, labels in pbar: 
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.criterion(output, labels) 
            loss.backward() 
            
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            running_loss += loss.item()
            _, prediction = torch.max(output,1)
            correct += prediction.eq(labels.view_as(prediction)).sum().item()
            total += labels.size(0)
            
            current_accuracy = 100 * correct/ total

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Accuracy":f"{current_accuracy:.2f}",
                "LR": f"{self.scheduler.get_last_lr()[0]:.6f}"
            })

            self.scheduler.step()

        return running_loss / len(self.train_loader), 100 * correct / total 
    
    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0 
        total = 0 

        pbar = tqdm(self.test_loader, desc="Evaluating")

        for images, labels in pbar:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            output = self.model(images)
            loss = self.criterion(output, labels)
            
            test_loss += loss.item()
            _,prediction = torch.max(output,1)
            correct += prediction.eq(labels.view_as(prediction)).sum().item()
            total += labels.size(0) 
            current_accuracy = 100 * correct / total 

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Accuracy":f"{current_accuracy:.2f}",
            })

        return test_loss / len(self.test_loader), 100 * correct / total 

    def train(self):
        best_accuracy = 0
        patience = 5 
        patience_counter = 0
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1} / {self.epochs}")
            
            train_loss, train_accuracy = self.train_epoch()
            
            self.history["train_losses"].append(train_loss)
            self.history["train_accuracies"].append(train_accuracy)

            test_loss, test_accuracy = self.evaluate()
            
            self.history["test_losses"].append(test_loss)
            self.history["test_accuracies"].append(test_accuracy)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                save_model(epoch=epoch,model=self.model,optimizer=self.optimizer,best_accuracy=best_accuracy,model_path=MODEL_PATH)
                patience_counter = 0 
            else:
                patience_counter += 1
            
            if patience_counter == patience:
                print(f"Early stopping triggerd after {epoch + 1} epochs")
                break
