import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from models.SwinUnet import SwinUNet
from models.UNet import UNet
from data.data_loader import MoNuSegDataset
from data.data_loader import BCSSDataset
from metrics.metrics import Metrics
import torch.nn as nn

def test(model_cat = 0, dataset = 0, vis = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iou_scores = []
    accuracy_scores = []
    f1_scores = []
    n_channels = 3
    
    if model_cat == 0:
        print("UNet MoNuSeg")
        # ./model_weights/unet_augment/monu/unet_monu_conv.pth
        # ./model_weights/unet_augment/monu/unet_monu_augment_fusion.pth
        # ./model_weights/unet_augment/monu/unet_monu_augment.pth
        n_classes = 1
        model = UNet(n_channels, n_classes, skip_type="convolutional").to(device)
        model_path = "./model_weights/unet_augment/monu/unet_monu_conv.pth"
    
    elif model_cat == 1:
        print("Swin Monuseg Model")
        print("fusion")
        # ./model_weights/swin_aug_models/monu/swinn_monu.pth        
        # ./model_weights/swin_aug_models/monu/swinn_monu_conv.pth
        # ./model_weights/swin_aug_models/monu/swinn_monu_fusion.pth
        n_classes = 1

        #fusion
        #model = SwinUNet(n_classes, device=device, skip_type="fusion").to(device)
        #model_path = "./model_weights/swin_aug_models/monu/swinn_monu_fusion.pth"

        #Default
        #model = SwinUNet(n_classes, device=device).to(device)
        #model_path = "./model_weights/swin_aug_models/monu/swinn_monu.pth"

        #Convolutional
        model = SwinUNet(n_classes, device=device, skip_type="convolutional").to(device)
        model_path = "./model_weights/swin_aug_models/monu/swinn_monu_conv.pth"
    
    elif model_cat == 2:
        print("Swin-Unet BCSS")
        # ./model_weights/swin_aug_models/bcss/swinn_bcss.pth
        # ./model_weights/swin_aug_models/bcss/swinn_bcss_conv.pth    
        # ./model_weights/swin_aug_models/bcss/swinn_bcss_fusion.pth
        n_classes = 3
        #model = SwinUNet(n_classes, device=device, skip_type="fusion").to(device)
        model = SwinUNet(n_classes, device=device).to(device)
        model_path = "./model_weights/swin_aug_models/bcss/swinn_bcss.pth"
    
    elif model_cat == 3:
        print("Unet BCSS")
        # ./model_weights/unet_augment/bcss/unet_bcss_augment.pth
        # ./model_weights/unet_augment/bcss/unet_bcss_augment_convolutional.pth
        #
        n_classes = 3
        model = UNet(n_channels, n_classes, skip_type="convolutional").to(device)
        model_path = "./model_weights/unet_augment/bcss/unet_bcss_augment_convolutional.pth"
    
    elif model_cat == 4:
        print("Unet BCSS512")
        # ./model_weights/unet_augment/bcss512/unet_bcss512_augment.pth
        # ./model_weights/unet_augment/bcss512/unet_bcss512_convolutional.pth
        #
        n_classes = 22
        model = UNet(n_channels, n_classes, skip_type="convolutional").to(device)
        model_path = "./model_weights/unet_augment/bcss512/unet_bcss512_convolutional.pth"
    else:
        print("Swin-Unet BCSS512")
        print("fusion")
        # ./model_weights/swin_aug_models/bcss512/swinn_bcss512.pth
        # ./model_weights/swin_aug_models/bcss512/swinn_bcss512_conv.pth
        # ./model_weights/swin_aug_models/bcss512/swinn_bcss512_fusion.pth
        n_classes = 22
        model = SwinUNet(n_classes, device=device, skip_type="fusion").to(device)
        model_path = "./model_weights/swin_aug_models/bcss512/swinn_bcss512_fusion.pth"

        #model = SwinUNet(n_classes, device=device).to(device)
        #model_path = "./model_weights/swin_aug_models/bcss512/swinn_bcss512.pth"
    
    # Load test dataset
    if dataset == 0:
        print("MoNuSeg")
        if vis == False:
            test_dataset = MoNuSegDataset(image_folder="./datasets/MoNuSeg/test/processed_images", 
                                        mask_folder="./datasets/MoNuSeg/test/processed_masks", device=device)
        else:
            test_dataset = MoNuSegDataset(image_folder="./datasets/MoNuSeg/test_vis/processed_image", 
                                        mask_folder="./datasets/MoNuSeg/test_vis/processed_mask", device=device)
    elif dataset == 1: 
        print("BCSS")
        if vis == False:
            test_dataset = BCSSDataset(image_dir="./datasets/BCSS/test", 
                                    mask_dir="./datasets/BCSS/test_mask", device=device)
        else:
            test_dataset = BCSSDataset(image_dir="./datasets/BCSS/test_vis", 
                                    mask_dir="./datasets/BCSS/test_vis_mask", device=device)
    else:
        print("BCSS 512")
        if vis == True:
            test_dataset = BCSSDataset(image_dir="./datasets/BCSS_512/test_images", 
                                       mask_dir="./datasets/BCSS_512/test_masks", device=device)
        else:
            test_dataset = BCSSDataset(image_dir="./datasets/BCSS_512/test_vis_image", 
                                       mask_dir="./datasets/BCSS_512/test_vis_mask", device=device)
                        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    class_labels = [
    'outside_roi', 'tumor', 'stroma', 'lymphocytic_infiltrate', 'necrosis_or_debris',
    'glandular_secretions', 'blood', 'exclude', 'metaplasia_NOS', 'fat',
    'plasma_cells', 'other_immune_infiltrate', 'mucoid_material', 'normal_acinus_or_duct',
    'lymphatics', 'undetermined', 'nerve', 'skin_adnexa', 'blood_vessel',
    'angioinvasion', 'dcis', 'other'
    ]

    # 22 visually distinct colors (feel free to change)
    class_colors = [
        '#000000', '#e6194b', '#3cb44b', '#ffe119', '#4363d8',
        '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
        '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
        '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075',
        '#808080', '#ffffff'
    ]

    # Create colormap
    cmap = ListedColormap(class_colors)

    # Run inference
    with torch.no_grad():
        for idx, (image, mask) in enumerate(test_loader):
            image = image.to(device)
            mask = mask.to(device)
            output = model(image)

            if model_cat in [0, 1]:
                iou_score = Metrics.calculate_iou(output, mask)
                acc = Metrics.acc(output, mask)
                f1 = Metrics.f1_score(output, mask)
    
            else:
                iou_score = Metrics.m_iou(output, mask, num_classes=n_classes)
                f1 = Metrics.m_f1(output, mask, num_classes=n_classes)
    
            iou_scores.append(iou_score)
            f1_scores.append(f1)

            if vis:
                img_np = image.squeeze().cpu().permute(1, 2, 0).numpy()
                mask_np = mask.squeeze().cpu().numpy()

                if n_classes == 1:
                    pred_np = output_binary.squeeze().cpu().numpy()
                else:
                    pred_np = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)

                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                im_pred = axes[0].imshow(img_np)
                im_pred_mask = axes[0].imshow(pred_np, cmap='gray' if n_classes == 1 else cmap, alpha=0.5)
                axes[0].set_title('Swin-UNet Prediction')
                axes[0].axis('off')
                # Ground Truth overlay
                im_gt = axes[1].imshow(img_np)
                im_gt_mask = axes[1].imshow(mask_np, cmap='jet' if n_classes == 1 else cmap, alpha=0.5)
                axes[1].set_title('Ground Truth Mask')
                axes[1].axis('off')
                # Create a legend using patches
                if n_classes > 1:
                    legend_patches = [
                        mpatches.Patch(color=class_colors[i], label=f"{i}: {class_labels[i]}")
                        for i in range(n_classes)
                    ]
                    # Add the legend to the right of the plots
                    plt.subplots_adjust(right=0.8)  # Adjust right margin to make space for the legend
                    fig.legend(handles=legend_patches, loc='center right', fontsize=8, title="Classes")
                plt.tight_layout(rect=[0, 0, 0.8, 1]) # Adjust tight layout to fit the legend.
                plt.show()

    # Compute final average scores
    avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    print(f"Final Average IoU on Test Set: {avg_iou:.4f}")
    print(f"Final Average F1 Score on Test Set: {avg_f1:.4f}")

def main():
    # Monuseg = 0, BCSS = 1, BCSS512 = 2
    print("Monu Default")
    test(model_cat = 3, dataset = 1, vis = True)

if __name__ == "__main__":
    main()