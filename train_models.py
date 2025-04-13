from data.data_loader import MoNuSegDataset
from data.data_loader import BCSSDataset

from train import Train_Models
from metrics.metrics import Metrics
from models.SwinUnet import SwinUNet
from models.UNet import UNet 

from torch.optim import Adam
from metrics.metrics import Metrics
from metrics.loss import DiceLoss
from tqdm import tqdm

import torch
from torch import Tensor, nn, optim
from torch.optim import Optimizer
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split


def train_model(model_type = "Unet_MoNuSeg"):
    '''
    This fucntion is responsible for configuring all of the models,
    and passing them into the training funciton. 
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Choose which model is going to be trained
    if model_type == "Unet_MoNuSeg":
        # The rest of the If statements are exaclty like this one, but use different
        # dataloaders, and loss functions 
        print("Unet_MoNuSeg_Fusion")
        train_ratio = 0.8
        val_ratio = 1 - train_ratio

        # Data loader
        train_dataset = MoNuSegDataset(image_folder="./datasets/MoNuSeg/train/processed_images",
                                       mask_folder="./datasets/MoNuSeg/train/processed_masks",
                                       augment=True, device=device)
        
        # Split up training dataset for train and validation
        total_size = len(train_dataset)
        train_size = int(train_ratio * total_size)
        val_size = total_size - train_size

        batch = 8
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(train_subset, batch_size=batch, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch, shuffle=False)

        # Initialize Model
        n_classes = 1
        n_channels = 3
        model = UNet(n_channels, n_classes, skip_type="convolutional").to(device)

        # Optimizer and Loss Function
        optimizer = Adam(model.parameters(), lr=1e-3)
        criterion1 = nn.BCEWithLogitsLoss()
        criterion2 = None

        # This will be the name for the save weights
        name = "unet_monu_conv"
        batch = 8
        epoch = 50

        # Metrics to measure the performance of the model during training
        iou_function=Metrics.calculate_iou
        pixel_accuracy_function=Metrics.acc

    elif model_type == "Swinn_MoNuSeg":
        print("Swinn_MoNuSeg")
        train_ratio = 0.8
        val_ratio = 1 - train_ratio

        train_dataset = MoNuSegDataset(image_folder="./datasets/MoNuSeg/train/processed_images", 
                                       mask_folder="./datasets/MoNuSeg/train/processed_masks",
                                       augment=True, device=device)
        total_size = len(train_dataset)
        train_size = int(train_ratio * total_size)
        val_size = total_size - train_size

        batch = 8
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(train_subset, batch_size=batch, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch, shuffle=False)

        n_classes = 1
        model = SwinUNet(n_classes, device=device).to(device)

        optimizer = Adam(model.parameters(), lr=1e-3)
        criterion1 = nn.BCEWithLogitsLoss()
        criterion2 = None

        name = "swinn_monu"
        epoch = 20
        iou_function=Metrics.calculate_iou
        pixel_accuracy_function=Metrics.acc

    elif model_type == "Swinn_BCSS":
        print("Swinn_BCSS_Conv")
        train_dataset = BCSSDataset(image_dir="./datasets/BCSS/train",
                                    mask_dir="./datasets/BCSS/train_mask",
                                    augment=True, device=device)
        
        val_dataset = BCSSDataset(image_dir="./datasets/BCSS/val",
                                  mask_dir="./datasets/BCSS/val_mask",
                                  augment=True, device=device)

        batch = 50
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch, shuffle=False)

        n_channels = 3
        n_classes = 3
        #model = SwinUNet(n_classes, device=device, skip_type="fusion").to(device)
        model = SwinUNet(n_classes, device=device, skip_type="convolutional").to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = DiceLoss().to(device)

        name = "swinn_bcss_conv"
        epoch = 10
        iou_function=Metrics.m_iou
        pixel_accuracy_function=Metrics.pixel_accuracy

    elif model_type == "Unet_BCSS":
        print("Unet_BCSS_convolutional!")
        train_dataset = BCSSDataset(image_dir="./datasets/BCSS/train", 
                                    mask_dir="./datasets/BCSS/train_mask", 
                                    augment=True, device=device)
        
        val_dataset = BCSSDataset(image_dir="./datasets/BCSS/val", 
                                  mask_dir="./datasets/BCSS/val_mask", 
                                  augment=True, device=device)

        batch = 50
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch, shuffle=False)

        n_channels = 3
        n_classes = 3
        model = UNet(n_channels, n_classes, skip_type="convolutional").to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = DiceLoss().to(device)

        name = "unet_bcss_augment"
        epoch = 10
        iou_function=Metrics.m_iou
        pixel_accuracy_function=Metrics.pixel_accuracy

    elif model_type == "Unet_BCSS_512":
        print("Unet_BCSS_512_fusion")
        train_dataset = BCSSDataset(image_dir="./datasets/BCSS_512/train_images", 
                                    mask_dir="./datasets/BCSS_512/train_masks", 
                                    augment=True, device=device)
        
        val_dataset = BCSSDataset(image_dir="./datasets/BCSS_512/val_images", 
                                  mask_dir="./datasets/BCSS_512/val_masks", 
                                  augment=True, device=device)

        batch = 50
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch, shuffle=False)

        n_channels = 3
        n_classes = 22
        model = UNet(n_channels, n_classes, skip_type="fusion").to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = DiceLoss().to(device)

        name = "unet_bcss_fusion_augment"
        epoch = 10
        iou_function=Metrics.m_iou
        pixel_accuracy_function=Metrics.pixel_accuracy
    else:
        print("Swinn_BCSS_512_Fusion")
        train_dataset = BCSSDataset(image_dir="./datasets/BCSS_512/train_images", 
                                    mask_dir="./datasets/BCSS_512/train_masks", 
                                    augment=True, device=device)
        
        val_dataset = BCSSDataset(image_dir="./datasets/BCSS_512/val_images", 
                                  mask_dir="./datasets/BCSS_512/val_masks", 
                                  augment=True, device=device)

        batch = 25
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch, shuffle=False)

        n_classes = 22
        #model = SwinUNet(n_classes, device=device, skip_type="fusion").to(device)
        #model = SwinUNet(n_classes, device=device, skip_type="convolutional").to(device)
        model = SwinUNet(n_classes, device=device).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = DiceLoss().to(device)

        name = "swinn_bcss512"
        epoch = 10
        iou_function=Metrics.m_iou
        pixel_accuracy_function=Metrics.pixel_accuracy

    # Send all of this configure variables to training function,
    # this will return the model weight and traning/validation stats
    trainer = Train_Models(
        name=name,
        model=model,
        batch=batch,
        epoch=epoch,
        optimizer=optimizer,
        criterion1=criterion1,
        criterion2=criterion2, 
        device=device,
        iou_function=iou_function,
        pixel_accuracy_function=pixel_accuracy_function,
    )

    trainer.train(train_loader, val_loader)

def main():
    # Train these three models
    train_model(model_type = "Swinn_BCSS")

    # Examples of other training calls
    #train_model(model_type = "Unet_MoNuSeg")
    #train_model(model_type = "Swinn_MoNuSeg")

if __name__ == "__main__":
    main()