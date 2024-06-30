# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:58:27 2024

@author: nikro
"""

import SimpleITK as sitk
import os
from skimage.restoration import denoise_tv_chambolle
import numpy as np

def apply_tv_denoising(image_array, weight):
    # Apply total variation denoising
    image_denoised = denoise_tv_chambolle(image_array, weight=weight)
    return image_denoised



def segment_image_with_otsu(input_image_path, output_image_path, number_of_thresholds=2, denoised_image_dir = r'path\to\save\the denoised\b900 images\if wanted'):
    # Load the input image
    image = sitk.ReadImage(input_image_path, sitk.sitkFloat32)
    image_array = sitk.GetArrayFromImage(image)

    # Apply TV denoising
    image_denoised_array = apply_tv_denoising(image_array, weight=5)
    image_denoised = sitk.GetImageFromArray(image_denoised_array.astype(np.float32))
    image_denoised.CopyInformation(image)

    # Optional: Save the denoised image
    denoised_image_filename = os.path.basename(input_image_path).replace('.nii.gz', '_denoised.nii.gz')
    denoised_image_path = os.path.join(denoised_image_dir, denoised_image_filename)
    sitk.WriteImage(image_denoised, denoised_image_path)

    

    # Normalize the image intensity to the range (0, 1)
    normalized_image = sitk.RescaleIntensity(image_denoised, 0, 1)

    # Apply Otsu multiple thresholds segmentation
    otsu_filter = sitk.OtsuMultipleThresholdsImageFilter()
    otsu_filter.SetNumberOfThresholds(number_of_thresholds)
    otsu_filter.SetNumberOfHistogramBins(512)
    segmented_image = otsu_filter.Execute(normalized_image)

    # Save the segmented image
    sitk.WriteImage(segmented_image, output_image_path)

def process_all_images(input_dir, output_dir):
    # Iterate over all directories in the input directory
    for folder in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, folder)):
            # Construct the path to the input image and output image
            input_image_path = os.path.join(input_dir, folder, f"{folder}.nii.gz")
            output_image_path = os.path.join(output_dir, f"otsu_{folder}.nii.gz")
            
            # Check if the input image file exists
            if os.path.isfile(input_image_path):
                # Apply segmentation and edge detection
                segment_image_with_otsu(input_image_path, output_image_path)
            else:
                print(f"Image not found for {folder}")

# Example usage
input_dir = r"path\to\DWI Sequence\b900 channel"
output_dir = r"path\to\desired\output"

process_all_images(input_dir, output_dir)
