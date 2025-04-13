import torch
from torch import Tensor
from torch.optim import Optimizer
from torch import nn
from torch.nn import functional as F
from torch import optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from models.SwinUnet import SwinUNet
from torch.optim import Adam

from metrics.metrics import Metrics
from metrics.loss import DiceLoss
from tqdm import tqdm

from data.data_loader import BCSSDataset

from models.UNet import UNet
from models.SwinUnet import SwinUNet

class Train_Models:
    def __init__(self, name, model, batch, epoch, optimizer, criterion1, criterion2=None, iou_function=None, pixel_accuracy_function=None, device="cpu"):
        self.name = name
        self.model = model
        self.batch = batch
        self.epoch = epoch
        self.optimizer = optimizer
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.iou_function = iou_function
        self.pixel_accuracy_function = pixel_accuracy_function

        if torch.cuda.is_available() and device == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def train(self, train_loader, val_loader):
        self.model = self.model.to(self.device)
    
        train_losses = []
        val_losses = []
        iou_scores_list = []
        pixel_accuracies_list = []
    
        for epoch in range(self.epoch):
            self.model.train()
            total_loss = 0
    
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epoch}", leave=False)
            for images, masks in progress_bar:
                images = images.to(self.device)
                masks = masks.to(self.device)
    
                outputs = self.model(images)
                loss1 = self.criterion1(outputs, masks)
    
                if self.criterion2 is not None:
                    loss2 = self.criterion2(outputs, masks)
                    loss = loss1 + loss2
                else:
                    loss = loss1
    
                total_loss += loss.item()
    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
                progress_bar.set_postfix(loss=loss.item())
    
            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
    
            self.model.eval()
            val_loss = 0
            iou_scores = []
            pixel_accuracies = []
    
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
    
                    outputs = self.model(images)
                    loss1 = self.criterion1(outputs, masks)
    
                    if self.criterion2 is not None:
                        loss2 = self.criterion2(outputs, masks)
                        loss = loss1 + loss2
                    else:
                        loss = loss1
    
                    val_loss += loss.item()
    
                    if self.iou_function:
                        iou = self.iou_function(outputs, masks)
                        iou_scores.append(iou)
                    if self.pixel_accuracy_function:
                        pixel_accuracy = self.pixel_accuracy_function(outputs, masks)
                        pixel_accuracies.append(pixel_accuracy)
    
            avg_val_loss = val_loss / len(val_loader)
            avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
            avg_pixel_accuracy = sum(pixel_accuracies) / len(pixel_accuracies) if pixel_accuracies else 0
    
            val_losses.append(avg_val_loss)
            iou_scores_list.append(avg_iou)
            pixel_accuracies_list.append(avg_pixel_accuracy)
    
            print(f"Epoch {epoch + 1}/{self.epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, IoU: {avg_iou:.4f}, Pixel Accuracy: {avg_pixel_accuracy:.4f}")
    
        print("Training complete!")
        torch.save(self.model.state_dict(), f"{self.name}.pth")
        print("Model saved.")
    
        # Save losses and metrics to a text file
        with open(f"{self.name}_training_results.txt", "w") as f:
            f.write("Epoch,Train Loss,Val Loss,IoU,Pixel Accuracy\n")
            for epoch in range(self.epoch):
                f.write(f"{epoch+1},{train_losses[epoch]:.6f},{val_losses[epoch]:.6f},{iou_scores_list[epoch]:.6f},{pixel_accuracies_list[epoch]:.6f}\n")
    
        print(f"Training results saved to {self.name}_training_results.txt")    


