import os
import re
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from sklearn.preprocessing import MinMaxScaler
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.svm import SVC
from joblib import dump
from skimage.restoration import denoise_tv_chambolle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
                
# Function to read NIFTI file using SimpleITK
def read_image(filepath):
    try:
        return sitk.ReadImage(filepath)
    except RuntimeError as e:
        logging.error(f"Error reading image {filepath}: {e}")
        return None

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
                    logging.warning(f"Permission denied for mask file: {mask_path}")
    return None

# Function to extract features using PyRadiomics
def extract_features(image, mask, full_spine_mask ):
    # Extract features from the lesion area
    lesion_features = extractor.execute(image, mask)
    exclude_keys = [
    'diagnostics_Versions_PyRadiomics', 
    'diagnostics_Versions_Numpy', 
    'diagnostics_Versions_SimpleITK', 
    'diagnostics_Versions_PyWavelet', 
    'diagnostics_Versions_Python', 
    'diagnostics_Configuration_Settings', 
    'diagnostics_Configuration_EnabledImageTypes', 
    'diagnostics_Image-original_Hash', 
    'diagnostics_Image-original_Dimensionality', 
    'diagnostics_Image-original_Spacing', 
    'diagnostics_Image-original_Size', 
    'diagnostics_Mask-original_Hash', 
    'diagnostics_Mask-original_Spacing', 
    'diagnostics_Mask-original_Size', 
    'diagnostics_Mask-original_BoundingBox', 
    'diagnostics_Mask-original_VoxelNum', 
    'diagnostics_Mask-original_VolumeNum', 
    'diagnostics_Mask-original_CenterOfMassIndex', 
    'diagnostics_Mask-original_CenterOfMass'
    ]
    lesion_features_filtered = {k: v for k, v in lesion_features.items() if k not in exclude_keys}

    # Initialize a dictionary for spine features
    spine_features = {}
    
    # If a full spine mask is provided, extract features from the entire spine
    if full_spine_mask is not None:
        full_spine_features = extractor.execute(image, full_spine_mask)
        spine_features = {k: v for k, v in full_spine_features.items() if k not in exclude_keys}
    
    # Combine both feature sets
    combined_features = {**lesion_features_filtered, **spine_features}
    print(f'Features: {combined_features}\n')
    
    return combined_features


# Function to process images for a single patient across all sequences and extract fused features
def process_patient_images(patient_id, sequence_dirs, mask_dir):
    features_by_sequence = {}

    for seq_dir in sequence_dirs:
        print(f'Processing sequence: {seq_dir}')
        sequence_name = os.path.basename(seq_dir)  # Get the name of the sequence from the directory path
        patient_dir_name = f'Patient_{patient_id}'
        image_filename = f'Patient_{patient_id}.nii.gz'
        patient_dir_path = os.path.join(seq_dir, patient_dir_name)
        image_path = os.path.join(patient_dir_path, image_filename)
        mask_path = find_corresponding_mask(patient_id, mask_dir)
        full_spine_mask = find_corresponding_mask(patient_id, full_spine_mask_dir)

        if os.path.exists(image_path) and mask_path and full_spine_mask:
            print(f"Processing image: {image_path}, mask: {mask_path}, full spine: {full_spine_mask}")
            image = read_image(image_path)
            
            image_array = sitk.GetArrayFromImage(image)

            # Apply TV denoising
            corrected_image_array = apply_tv_denoising(image_array, weight=5)
            corrected_image = sitk.GetImageFromArray(corrected_image_array.astype(np.float32))
            corrected_image.CopyInformation(image)
            
            mask = read_image(mask_path)
            spine = read_image(full_spine_mask)
            features = extract_features(corrected_image, mask, spine)  # This should return a feature vector
            features_by_sequence[sequence_name] = features
        else:
            print(f"Missing image or mask for patient {patient_id} in sequence {seq_dir}")
    
    return features_by_sequence

# Initialize PyRadiomics feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.settings['normalize'] = True
extractor.settings['normalizeScale'] = 1  # Normalize the image intensity to 0-255
#extractor.settings['label'] = 255

# Main script execution
if __name__ == '__main__':
    mask_dir = r"path\to\lesion\masks"
    full_spine_mask_dir = r'path\to\full_spine masks'
    sequence_dirs = [
        r'D:\Slicer 5.6.1\Feature extraction\T1W',
        r'D:\Slicer 5.6.1\Feature extraction\T2W',
        r'D:\Slicer 5.6.1\Feature extraction\T2DIXON_IP',
        r'D:\Slicer 5.6.1\Feature extraction\T1DIXON_IP',
        r'D:\Slicer 5.6.1\Feature extraction\b900',
        r'D:\Slicer 5.6.1\Feature extraction\b50'
    ]
    
    myeloma_label = 1
    pre_myeloma_label = 0
    all_fused_features = []

    y = []

    # Get a list of all patient IDs from all sequences
    # Replace the patient_ids collection logic with the new function
    patient_ids = get_sorted_unique_patient_ids(sequence_dirs)
    print(f'Patient ids: {patient_ids}')

    all_patient_features = {}
    for patient_id in patient_ids:
        patient_features = process_patient_images(patient_id, sequence_dirs, mask_dir)
        all_patient_features[patient_id] = patient_features
        # Assign label based on patient ID
        label = myeloma_label if patient_id <= 28 else pre_myeloma_label
        y.append(label)

    # Determine the dimensions
    num_patients = len(patient_ids)
    num_sequences = len(sequence_dirs)
    num_features = len(next(iter(next(iter(all_patient_features.values())).values())))  # Get feature count from the first item

    # Initialize the 3D array
    feature_matrix = np.full((num_patients, num_sequences, num_features), np.nan)

    # Assuming feature_matrix is initialized as shown previously
    for i, patient_id in enumerate(patient_ids):
        for j, seq_dir in enumerate(sequence_dirs):
            sequence_name = os.path.basename(seq_dir)
            if patient_id in all_patient_features and sequence_name in all_patient_features[patient_id]:
                features_dict = all_patient_features[patient_id][sequence_name]
                for k, feature_name in enumerate(sorted(features_dict.keys())):
                    feature_value = features_dict[feature_name]
                    # Check if the feature value is an array with a single value and convert
                    if isinstance(feature_value, np.ndarray) and feature_value.size == 1:
                        feature_matrix[i, j, k] = feature_value.item()
                    elif feature_value is not None:  # Assuming None or another marker indicates available data
                        feature_matrix[i, j, k] = feature_value
    
    
    # Flattening the 3D matrix into a 2D matrix
    flattened_feature_matrix = feature_matrix.reshape((num_patients, num_sequences * num_features))
    # Define the preprocessing and feature selection steps
    preprocessing_steps = [
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('variance_threshold', VarianceThreshold(threshold=0)),
        ('feature_selection', SelectKBest(f_classif, k='all'))
        ]

    # Define the SVM and grid search parameters
    svm_classifier = SVC()
    param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'svm__gamma': ['scale', 'auto']
    }

    # Create the pipeline
    pipeline_steps = preprocessing_steps + [('svm', svm_classifier)]
    pipeline = Pipeline(steps=pipeline_steps)

    # Step 3: Train the Model with Grid Search
    X_train, X_test, y_train, y_test = train_test_split(flattened_feature_matrix, y, test_size=0.2, random_state=42)
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Step 4: Save the Best Pipeline
    best_pipeline = grid_search.best_estimator_
    dump(best_pipeline, 'best_SVM_full_dataset.joblib')

    # Evaluation
    y_pred = best_pipeline.predict(X_test)
    print("Best parameters found:", grid_search.best_params_)
    print("Classification report for the best classifier:")
    print(classification_report(y_test, y_pred))
    print("Accuracy score for the best classifier:", accuracy_score(y_test, y_pred))