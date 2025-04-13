
BELOW ARE THE SUMMARIES FOR ALL OF THE SCRIPTS USED IN THIS PROJECT

Data
 -augment.py:
    This scirpt is responsible for augmenting each image during training. I augmentations used horizontal/vertical flips, 
    rotation, elastic transform, grid distortion, and changes in brightness. 

 -data_loader.py:
    This script is responsible for loading the images for boths models during testing. 
    The dataloader for BCSS came standard in the challange, so I used the one that was provided, you can find the implmentatation here:
        https://www.kaggle.com/code/whats2000/u-net-bcss-segmentation

 -data_preprocessing.py
    This script was used during preprocessing for the monuseg and bcss512 dataset. Here you'll find that I resized the image 
    and if needed divided evenly so it would match the input size for the images.

Datasets
    -BCSS and BCSS512: 
    Below is the link for where I download the dataset from
    https://www.kaggle.com/datasets/whats2000/breast-cancer-semantic-segmentation-bcss/data

    -MoNuSeg
    Below is the link for where I download the dataset from
    https://monuseg.grand-challenge.org/Data/

Metrics
    -loss.py:
    This funciton has DiceLoss class that I used for both of my models. 
    I found a couple github repositories online that had to code here is an example:
        https://www.kaggle.com/code/whats2000/u-net-bcss-segmentation
    
    -metrics.py:
    This scirpt measures the performance of the models during and after training.
    Here, you will find the mIoU and F1 score. The BCSS challenge provides official code to accurately measure 
    the performance of contestants' models. Here is the provided example:
        https://www.kaggle.com/code/whats2000/u-net-bcss-segmentation

Models
    - SwinUnet.py:
    This script has the code for the SwinUnet model, since I did not want to train the model from scratch I used a 
    pretrain swin encoder. Here is the paper I based my architecture off of, it doesnt not match perfectly because of the
    pretrained encoder.
        https://arxiv.org/pdf/2105.05537

    - Unet.py:
    This code as the standard pytorch implementation for UNet architecture. Again, I used code from this souce, 
    only thing I modified in this code was the skip connection layers which my paper is about. 
        https://www.kaggle.com/code/whats2000/u-net-bcss-segmentation

train_models.py:
    - This script that builds all the models and training detatil and calls train.py

train.py
    - Training loop for all the model combinations

train_models.py
    - Testing loop for all the model combintations
