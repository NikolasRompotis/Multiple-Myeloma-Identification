# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 18:36:01 2024

@author: nikro
"""

import os
import re
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.restoration import denoise_tv_chambolle

def load_mri_image_sitk(file_path):
    sitk_image = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(sitk_image)
    corrected_image_array = apply_tv_denoising(array, weight=5)
    corrected_image = sitk.GetImageFromArray(corrected_image_array.astype(np.float32))
    corrected_image.CopyInformation(sitk_image)
    return corrected_image_array, corrected_image

def apply_tv_denoising(image_array, weight):
    # Apply total variation denoising
    image_denoised = denoise_tv_chambolle(image_array, weight=weight)
    return image_denoised

# Function to extract numerical ID from filename
def extract_numerical_id(filename):
    match = re.search(r'\d+', filename)
    if match:
        
        return int(match.group())
    return None

def get_sorted_unique_patient_ids(sequence_dirs):
    patient_ids = set()
    for seq_dir in sequence_dirs:
        for filename in os.listdir(seq_dir):
            patient_id = extract_numerical_id(filename)
            if patient_id is not None:
                patient_ids.add(patient_id)
    # Convert to list, sort numerically, and return
    sorted_patient_ids = sorted(list(patient_ids))
    return sorted_patient_ids

# Function to find the corresponding mask file for the patient image
def find_corresponding_mask(patient_id, mask_dir):
    patient_id_int = int(patient_id)
    for mask_filename in os.listdir(mask_dir):
        if mask_filename.endswith('.nii.gz'):  # Ensure the file is a NIFTI image
            mask_id = extract_numerical_id(mask_filename)
            if mask_id == patient_id_int:
                mask_path = os.path.join(mask_dir, mask_filename)
                if os.access(mask_path, os.R_OK):  # Check if the file is readable
                    return mask_path
                else:
                    print(f"Permission denied for mask file: {mask_path}")
    return None

def calculate_adc_map(array_s1, array_s2, b1, b2):
    epsilon = 1e-9
    adc_map = np.log((array_s2 + epsilon) / (array_s1 + epsilon)) / (b1 - b2)
    
    return adc_map

def segment_adc_map(adc_map, adc_min_value, adc_max_value):
    adc_segmented = np.logical_and(adc_map >= adc_min_value, adc_map <= adc_max_value)
    
    return adc_segmented.astype(np.float32)

def post_process_segmented_image(adc_segmented):
    kernel_size = 3
    adc_segmented_closed = ndimage.binary_closing(adc_segmented, structure=np.ones((kernel_size, kernel_size, kernel_size)))
    adc_segmented_smooth = ndimage.median_filter(adc_segmented_closed, size=3)
    return adc_segmented_smooth

def save_segmented_image_sitk(segmented_image, reference_image, output_path):
    segmented_image_sitk = sitk.GetImageFromArray(segmented_image.astype(np.float32))
    segmented_image_sitk.CopyInformation(reference_image)
    sitk.WriteImage(segmented_image_sitk, output_path)

def process_adc_segmentation(path_to_s1_image, path_to_s2_image, mask_path, b1, b2, adc_min_value, adc_max_value, output_path):
    array_s1, sitk_image_s1 = load_mri_image_sitk(path_to_s1_image)
    array_s2, sitk_image_s2 = load_mri_image_sitk(path_to_s2_image)
    mask_image = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask_image)
    
    
        
    adc_map = calculate_adc_map(array_s1, array_s2, b1, b2)
    save_segmented_image_sitk(adc_map, sitk_image_s1, output_path)
    print('Saved ADC map\n')
    # Initially, set all pixels outside the mask (mask != 2) to 0
    outside_mask = mask_array != 2
    adc_map[outside_mask] = 0
    # Isolate the area where mask == 2
    
    mask_area = mask_array == 2
    
    # Calculate the mean value of the pixels in the masked area
    masked_pixels = adc_map[mask_area]
    
    mean_val = masked_pixels.mean()
    
    # Check if the mean value falls within the specified ADC range
    if adc_min_value <= mean_val <= adc_max_value:
        
        # Set pixels within the ADC range and mask to 1
        adc_map[mask_area] = 1
    else:
        # Set pixels outside the ADC range but within the mask to 0
        
        adc_map[mask_area] = 0
   
    
    #adc_segmented = segment_adc_map(adc_map, adc_min_value, adc_max_value)
    #adc_segmented_smooth = post_process_segmented_image(adc_segmented)
    
    save_segmented_image_sitk(adc_map, sitk_image_s2, output_path)
    print('Saved ADC mask\n')
    slice_index = adc_map.shape[0] // 2
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(adc_map[slice_index, :, :], cmap='gray')
    plt.title('ADC Map')
    plt.subplot(1, 2, 2)
    plt.imshow(adc_map[slice_index, :, :], cmap='gray')
    plt.title('ADC Map')
    plt.show()

def process_adc_segmentation_for_multiple_patients(base_dir, b1, b2, adc_min_value, adc_max_value):
    s1_dir = os.path.join(base_dir, "Feature extraction", "b50")
    s2_dir = os.path.join(base_dir, "Feature extraction", "b900")
    mask_dir = os.path.join(base_dir, "Lesion_Iso", "Otsu_b900")
    output_dir = os.path.join(base_dir, "Lesion_Iso", "ADC")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get sorted unique patient IDs from s1 and s2 directories
    patient_ids = get_sorted_unique_patient_ids([s1_dir, s2_dir])

    for patient_id in patient_ids:
        path_to_s1_image = os.path.join(s1_dir, f"Patient_{patient_id}", f"Patient_{patient_id}.nii.gz")
        path_to_s2_image = os.path.join(s2_dir, f"Patient_{patient_id}", f"Patient_{patient_id}.nii.gz")
        mask_path = find_corresponding_mask(str(patient_id), mask_dir)
        output_path = os.path.join(output_dir, f"ADC_{patient_id}.nii.gz")

        if mask_path:
            print(f"Processing Patient_{patient_id}")
            process_adc_segmentation(path_to_s1_image, path_to_s2_image, mask_path, b1, b2, adc_min_value, adc_max_value, output_path)
            print(f"Finished processing Patient_{patient_id}\n")
        else:
            print(f"Mask not found for Patient_{patient_id}")
# Specify your file paths and parameters
base_dir = r"D:\Slicer 5.6.1"
b1, b2 = 50, 900
adc_min_value, adc_max_value = 0.6*10**-3, 1.4*10**-3

# Execute the process
process_adc_segmentation_for_multiple_patients(base_dir, b1, b2, adc_min_value, adc_max_value)
