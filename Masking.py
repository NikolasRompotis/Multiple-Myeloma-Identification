# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 18:04:16 2024

@author: nikro
"""
import SimpleITK as sitk
import os
import re

def extract_numerical_id(filename):
    """Extract numerical identifier from filenames."""
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return None

def read_image(filepath):
    """Read a NIFTI file using SimpleITK."""
    return sitk.ReadImage(filepath)

def apply_mask(image, mask):
    """Apply the mask to the MRI image. Both `image` and `mask` are SimpleITK images."""
    masked_image = sitk.Mask(image, mask)
    return masked_image

def save_image(image, directory, filename):
    """Saves the masked image to the specified directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    sitk.WriteImage(image, os.path.join(directory, filename))

def process_patient_images(patient_folder, mask_dir, output_dir):
    """Process images for a single patient."""
    patient_id = extract_numerical_id(patient_folder)
    patient_image_dir = os.path.join(image_dir, patient_folder)
    
    # Check if patient_image_dir is actually a directory
    if not os.path.isdir(patient_image_dir):
        print(f"Skipping {patient_image_dir} because it's not a directory.")
        return
    
    image_files = [f for f in os.listdir(patient_image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    mask_files = [f for f in os.listdir(mask_dir) if extract_numerical_id(f) == patient_id]

    for image_filename in image_files:
        for mask_filename in mask_files:
            image_filepath = os.path.join(patient_image_dir, image_filename)
            mask_filepath = os.path.join(mask_dir, mask_filename)

            image = read_image(image_filepath)
            mask = read_image(mask_filepath)
            masked_image = apply_mask(image, mask)
            patient_output_dir = os.path.join(output_dir, patient_folder)
            save_image(masked_image, patient_output_dir, f'Patient_{patient_id}.nii.gz')

if __name__ == '__main__':
    image_dir = r'path\to\registered\sequence\and channel if available'  # Update this path for images
    mask_dir = r'path\to masks\from the U-net'  # Update this path for masks
    output_dir = r'path\to\desired\output'  # Update this path for output
    patient_folders = [f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f)) and f.startswith('Patient_')]
    for patient_folder in patient_folders:
        process_patient_images(patient_folder, mask_dir, output_dir)

