import torch
import torch.nn as nn
from torchvision import models

class FishQualityClassifier(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # Add dropout for uncertainty estimation
        self.dropout = nn.Dropout(p=0.3)

        # Replace final layer → 1 logit
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False

    def forward(self, x):
        features = self.model.avgpool(self.model.layer4(self.model.layer3(
                     self.model.layer2(self.model.layer1(self.model.relu(
                     self.model.bn1(self.model.conv1(x))))))))
        
        features = torch.flatten(features, 1)
        features = self.dropout(features)  # MC dropout active in eval for uncertainty
        logit = self.model.fc(features)
        return logit  # raw logits (not sigmoid)