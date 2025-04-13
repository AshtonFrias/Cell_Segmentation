import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, eps=1e-6):
        # Apply softmax to the inputs to convert logits to probabilities
        inputs = torch.softmax(inputs, dim=1)

        # Create the one-hot encoding of targets for multi-class compatibility
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # Reshape to match input tensor shape

        # Flatten the inputs and targets for element-wise operations
        inputs_flat = inputs.view(inputs.shape[0], inputs.shape[1], -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)

        # Calculate intersection and union for each class
        intersection = torch.sum(inputs_flat * targets_flat, dim=2)
        total = torch.sum(inputs_flat, dim=2) + torch.sum(targets_flat, dim=2)

        # Calculate dice score for each class and then average across classes
        dice = (2. * intersection + eps) / (total + eps)  # Add eps for numerical stability
        dice = dice.mean(dim=1)  # Average over classes

        # Dice loss is 1 minus the dice score
        return 1.0 - dice.mean()
