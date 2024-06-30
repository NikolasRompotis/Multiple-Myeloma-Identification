import os
import SimpleITK as sitk

def process_mri_image(image_path, mask_path, output_dir):
    # Load the image and mask using SimpleITK
    image = sitk.ReadImage(image_path, sitk.sitkFloat32)
    mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    
    # Modify the mask to focus only on voxels with value 2
    modified_mask = sitk.BinaryThreshold(mask, lowerThreshold=2, upperThreshold=2, insideValue=1, outsideValue=0)
    
    # Apply the mask to the image
    masked_image = sitk.Mask(image, modified_mask)
    
    # Convert masked image to a binary image suitable for ConnectedComponent
    binary_masked_image = sitk.NotEqual(masked_image, 0)

    # Find connected components in the binary masked image
    connected_components = sitk.ConnectedComponent(binary_masked_image)
    
    # Relabel components to ensure components are sorted by size (largest to smallest)
    relabeled_components = sitk.RelabelComponent(connected_components, sortByObjectSize=True)
    
    # Count the number of connected components
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(relabeled_components)
    number_of_components = label_stats.GetNumberOfLabels()
    print(f"Number of connected components: {number_of_components}")
    
    # Prepare the output image based on the input dimensions and type
    output_image = sitk.Image(masked_image.GetSize(), sitk.sitkUInt8)
    output_image.CopyInformation(masked_image)
    
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(relabeled_components, masked_image)

    # Iterate through each component to calculate average and apply threshold
    for label_idx in range(1, number_of_components + 1):
        print(f"Processing label {label_idx}")
        
        # Extract the single component
        single_component = sitk.Equal(relabeled_components, label_idx)
        
        if label_stats.GetNumberOfPixels(label_idx) > 0:
            # Calculate average voxel intensity within the component
            average_intensity = stats.GetMean(label_idx)
            print(f"Average intensity for label {label_idx}: {average_intensity}")
            
            # Check the average intensity and modify the output image accordingly
            if 0.6e-3 <= average_intensity <= 1.4e-3:
                print(f"Label {label_idx} meets criteria, modifying output image.")
                output_image = sitk.Add(output_image, sitk.Cast(single_component, sitk.sitkUInt8))
        else:
            print(f"Label {label_idx} has no pixels, skipping.")
    
    # Save the final image
    output_image_path = os.path.join(output_dir, os.path.basename(image_path).replace('.nii.gz', '_mask.nii.gz'))
    sitk.WriteImage(output_image, output_image_path)
    print(f"Processing complete. Image saved to {output_image_path}")

def process_all_images_and_masks(image_dir, mask_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List all images and masks
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])
    
    # Ensure the number of images and masks match
    if len(image_files) != len(mask_files):
        print("The number of images and masks do not match.")
        return
    
    # Process each image/mask pair
    for image_file, mask_file in zip(image_files, mask_files):
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)
        print(f"Processing {image_file} and {mask_file}")
        process_mri_image(image_path, mask_path, output_dir)

# Specify directories
image_dir = r'path\to\ADC maps'
mask_dir = r'path\to\otsu created\lesion masks'
output_dir = r'desired\output'

# Process all images and masks
process_all_images_and_masks(image_dir, mask_dir, output_dir)
