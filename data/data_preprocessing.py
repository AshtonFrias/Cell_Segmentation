import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import glob
from PIL import Image
import shutil
import random

def save_mask_to_txt(mask, txt_path):
    """Saves a grayscale mask to a text file."""
    np.savetxt(txt_path, mask, fmt='%d')

class MoNuSegPreprocessor:
    def __init__(self, image_folder="./datasets/MoNuSeg/TrainingData/Images", 
                mask_folder="./datasets/MoNuSeg/TrainingData/Annotations", 
                output_image_folder="./processed_images", 
                output_mask_folder="./processed_masks", 
                target_size=(224, 224), sub_image_size=(224, 224)):
                
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.output_image_folder = output_image_folder
        self.output_mask_folder = output_mask_folder
        self.target_size = target_size
        self.sub_image_size = sub_image_size

        # Create output directories if they don't exist
        Path(self.output_image_folder).mkdir(parents=True, exist_ok=True)
        Path(self.output_mask_folder).mkdir(parents=True, exist_ok=True)

        # Get sorted lists of image and annotation files
        self.image_files = sorted([f for f in os.listdir(self.image_folder) if f.endswith(".tif")])
        self.xml_files = sorted([f for f in os.listdir(self.mask_folder) if f.endswith(".xml")])

        print("Found Images:", self.image_files)
        print("Found Annotations:", self.xml_files)

        # Check for missing data
        if not self.image_files or not self.xml_files:
            raise ValueError("No images or annotations found! Check dataset paths and file extensions.")

    def parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        contours = []
        for region in root.findall(".//Region"):
            points = []
            for vertex in region.findall(".//Vertex"):
                x, y = float(vertex.get("X")), float(vertex.get("Y"))
                points.append((x, y))  # Keep as floats
            if points:
                contours.append(np.array(points, dtype=np.float32))  # Keep as float32
        return contours

    def resize_image(self, image, new_height=896, new_width=896):
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image

    def split_image(self, image):
        sub_images = []
        height, width = image.shape[:2]

        print(f"Original image size: {height}x{width}")

        # Ensure padding is applied only if necessary
        pad_bottom = (self.sub_image_size[0] - height % self.sub_image_size[0]) if height % self.sub_image_size[0] != 0 else 0
        pad_right = (self.sub_image_size[1] - width % self.sub_image_size[1]) if width % self.sub_image_size[1] != 0 else 0
        image_padded = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        height, width = image_padded.shape[:2]
        stride = 224  # Ensure we step by 224 pixels, no overlap

        for i in range(0, height, stride):
            for j in range(0, width, stride):
                print(f"Creating sub-image at: ({i}, {j}) → ({i + 224}, {j + 224})")
                sub_image = image_padded[i:i + self.sub_image_size[0], j:j + self.sub_image_size[1]]
                sub_images.append(sub_image)

        return sub_images

    def create_mask(self, contours, image_shape):
        mask = np.zeros(image_shape, dtype=np.uint8)

        for contour in contours:
            contour_int = np.round(contour).astype(np.int32)
            cv2.drawContours(mask, [contour_int], -1, 255, cv2.FILLED)

        mask[mask > 0] = 255 

        return mask

    def process(self):
        count = 0
        for image_file, xml_file in zip(self.image_files, self.xml_files):
            # Load the image and annotation
            image_path = os.path.join(self.image_folder, image_file)
            xml_path = os.path.join(self.mask_folder, xml_file)

            image = cv2.imread(image_path)
            contours = self.parse_xml(xml_path)
            mask = self.create_mask(contours, image.shape)

            if image is None or mask is None:
                print(f"Failed to load {image_file} or {xml_file}. Skipping...")
                continue

            # Resize image and mask to 896x896 before splitting
            image_resized = self.resize_image(image)
            mask_resized = self.resize_image(mask)

            # Split the resized image and mask into patches
            image_patches = self.split_image(image_resized)
            mask_patches = self.split_image(mask_resized)

            # Save each image and mask patch
            for i, (img_patch, msk_patch) in enumerate(zip(image_patches, mask_patches)):
                 msk_patch = msk_patch.astype(np.uint8)
                 output_image_path = os.path.join(self.output_image_folder, f"{os.path.splitext(image_file)[0]}_patch_{i}.png")
                 output_mask_path = os.path.join(self.output_mask_folder, f"{os.path.splitext(xml_file)[0]}_patch_{i}.png")

                 cv2.imwrite(output_image_path, img_patch)
                 cv2.imwrite(output_mask_path, msk_patch)

            print(f"Processed and saved patches for {image_file} and {xml_file}")

class BCSS512Preprocessor:
    def __init__(self, image_dir, mask_dir, output_image_dir, output_mask_dir, target_size=(224, 224)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.output_image_dir = output_image_dir
        self.output_mask_dir = output_mask_dir
        self.target_size = target_size

        # Create output directories if they don't exist
        os.makedirs(self.output_image_dir, exist_ok=True)
        os.makedirs(self.output_mask_dir, exist_ok=True)

    def resize_and_save(self, image_path, mask_path):
        # Load the image and mask
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # Resize both the image and mask to target size
        image_resized = image.resize(self.target_size, Image.Resampling.LANCZOS)
        mask_resized = mask.resize(self.target_size, Image.Resampling.NEAREST)

        # Extract filenames from paths
        image_filename = os.path.basename(image_path)
        mask_filename = os.path.basename(mask_path)

        # Save the resized image and mask to output directories
        image_resized.save(os.path.join(self.output_image_dir, image_filename))
        mask_resized.save(os.path.join(self.output_mask_dir, mask_filename))

    def process(self):
        # Get list of image and mask files
        image_files = glob.glob(os.path.join(self.image_dir, '*.png')) 
        mask_files = glob.glob(os.path.join(self.mask_dir, '*.png'))

        # Process each image and corresponding mask
        for image_file in image_files:
            # Find the corresponding mask for each image
            mask_file = os.path.join(self.mask_dir, os.path.basename(image_file))
            
            if mask_file in mask_files:
                self.resize_and_save(image_file, mask_file)
            else:
                print(f"Warning: No corresponding mask found for {image_file}")    

def main():
    # File Path
    output_image_folder = "./datasets/BCSS_512/val_images"
    output_mask_folder = "./datasets/BCSS_512/val_masks"
    image_folder = "./datasets/BCSS_512/val_512"
    mask_folder = "./datasets/BCSS_512/val_mask_512"
    size_choice = 224

    preprocessor = BCSS512Preprocessor(image_folder, mask_folder, 
                                       output_image_folder, output_mask_folder)
    preprocessor.process()

    '''
    preprocessor = MoNuSegPreprocessor(image_folder, mask_folder, output_image_folder, 
                                       output_mask_folder, target_size=target_size, 
                                       sub_image_size=sub_image_size)
    preprocessor.process()
    '''
if __name__ == "__main__":
    main()
