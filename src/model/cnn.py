import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self, dropout_rate: float = 0.3) -> None:
        super().__init__()

        self.features = nn.Sequential(
            #first feature detector layer 
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #second layer
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #third layer 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout2d(p=dropout_rate * 0.5),

            nn.AdaptiveAvgPool2d(output_size=((4,4)))
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=128 * 4 * 4, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, input_data: torch.Tensor):
        output = self.features(input_data)
        output = torch.flatten(output, 1)
        output = self.classifier(output)

        return output
