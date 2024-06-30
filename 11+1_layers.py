import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import torchio as tio
import scipy.ndimage



# Define the 3D U-Net model
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm3d(out_channels)
        )
        self.residual = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1),  # 1x1 convolution
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        return self.conv(x) + self.residual(x)

class UNet3D(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder layers (downsampling)
        self.layer1 = DoubleConv(1, 32)
        self.layer2 = DoubleConv(32, 64)
        self.layer3 = DoubleConv(64, 128)
        self.layer4 = DoubleConv(128, 256)
        self.layer5 = DoubleConv(256, 512)
        self.layer6 = DoubleConv(512, 1024)  # New layer added to reach 1024 filters

        # Decoder layers (upsampling)
        self.layer7 = DoubleConv(1024 + 512, 512)
        self.layer8 = DoubleConv(512 + 256, 256)
        self.layer9 = DoubleConv(256 + 128, 128)
        self.layer10 = DoubleConv(128 + 64, 64)
        self.layer11 = DoubleConv(64 + 32, 32)
        self.layer12 = torch.nn.Conv3d(32, 1, 1)  # Final convolution to get the output

        # Pooling and upsample
        self.maxpool = torch.nn.MaxPool3d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

    def forward(self, x):
        print(f"Input size: {x.size()}")

        # Encoder path
        x1 = self.layer1(x)
       
        x1m = self.maxpool(x1)
        

        x2 = self.layer2(x1m)
        x2m = self.maxpool(x2)

        x3 = self.layer3(x2m)
        x3m = self.maxpool(x3)

        x4 = self.layer4(x3m)
        x4m = self.maxpool(x4)
      

        x5 = self.layer5(x4m)
        x5m = self.maxpool(x5)

        x6 = self.layer6(x5m)

        # Decoder path
        x7 = self.upsample(x6)
        x7 = torch.cat([x7, x5], dim=1)
        x7 = self.layer7(x7)

        x8 = self.upsample(x7)
        x8 = torch.cat([x8, x4], dim=1)
        x8 = self.layer8(x8)

        x9 = self.upsample(x8)
        x9 = torch.cat([x9, x3], dim=1)
        x9 = self.layer9(x9)

        x10 = self.upsample(x9)
        x10 = torch.cat([x10, x2], dim=1)
        x10 = self.layer10(x10)

        x11 = self.upsample(x10)
        x11 = torch.cat([x11, x1], dim=1)
        x11 = self.layer11(x11)

        output = self.layer12(x11)
        print(f"Output size: {output.size()}")

        return output



class NiftiDataset(Dataset):
    def __init__(self, folder_path, output_size=(256, 512, 32), is_train=True, indices=None):
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith('BOT') and f.endswith('.gz')]
        self.mask_paths = [p.replace('BOT', 'ROB') for p in self.image_paths]
        self.output_size = output_size
        self.is_train = is_train
        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]
            self.mask_paths = [self.mask_paths[i] for i in indices]

    def calculate_target_spacing(self, image_path):
        sample_image = nib.load(image_path)
        original_shape = np.array(sample_image.shape)
        original_spacing = np.array(sample_image.header.get_zooms())

        scale_factor = original_shape / np.array(self.output_size)
        
        # Calculate target spacing and round to avoid small discrepancies
        target_spacing = np.round(original_spacing * scale_factor, decimals=2)
        

        return target_spacing
        
    def print_file_names(self):
        print("Dataset file names:")
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            print(f"Image: {img_path}, Mask: {mask_path}")
        print(f"Total number of files: {len(self.image_paths)}")
        
    def get_file_names(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        return image_path, mask_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        target_spacing = (1, 1, 1)

        base_transforms = [
            
            tio.Resample(target_spacing),
            
            tio.CropOrPad(self.output_size),  # Ensure same spatial shape
            tio.ZNormalization()
        ]
        if self.is_train:
            augmentation_transforms = [
                #tio.RandomAffine(
                    #scales=(0.9, 1.1),
                    #degrees=10,
                    #translation=5
                #),
                tio.RandomElasticDeformation(
                    num_control_points=8,
                    max_displacement=5
                ),
                tio.RandomFlip(axes=(0,)),  # Randomly flip along different axes
                tio.RandomBiasField(),  # Simulate MRI bias field artifacts
                tio.RandomNoise(),  # Add random noise
                tio.RandomBlur(),  # Add random blurring
                tio.RandomGamma()  # Adjust gamma values for intensity variation
            ]
            transforms = tio.Compose(base_transforms + augmentation_transforms)
        else:
            transforms = tio.Compose(base_transforms)

        subject = tio.Subject(image=tio.ScalarImage(image_path), mask=tio.LabelMap(mask_path))
        transformed_subject = transforms(subject)

        return {
            'image': transformed_subject['image'][tio.DATA].float(),
            'mask': transformed_subject['mask'][tio.DATA].float()
        }

def Dice_loss(pred, target, smooth = 1.):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    dice = 1 - (2.*intersection + smooth)/(pred.sum() + target.sum() + smooth)
    return dice
    
    
class CombinedDiceBCELoss(nn.Module):
    def __init__(self):
        super(CombinedDiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        dice = Dice_loss(inputs, targets)
        bce = self.bce(inputs, targets)
        combined_loss = 0.7*dice + 0.3*bce
        return combined_loss, dice, bce
        
        

        
def print_dataloader_file_names(dataloader, dataset):
    for batch_idx, _ in enumerate(dataloader):
        image_path, mask_path = dataset.get_file_names(batch_idx)
        print(f"Batch {batch_idx + 1}: Image: {image_path}, Mask: {mask_path}")

# Calculate Metrics (Dice/IoU)
def calculate_dice_iou(outputs, labels):
    # Binarize predictions and labels
    outputs = outputs > 0.5
    labels = labels > 0.5

    # Flatten tensors to calculate global metrics
    outputs_flat = outputs.view(-1)
    labels_flat = labels.view(-1)

    # Calculate TP, FP, FN for Dice and IoU
    intersection = (outputs_flat & labels_flat).sum()
    total = (outputs_flat | labels_flat).sum()
    union = total - intersection
    TP = intersection
    FP = (outputs_flat & ~labels_flat).sum()
    FN = (~outputs_flat & labels_flat).sum()

    # Calculate Dice Score
    dice = (2 * TP).float() / (2 * TP + FP + FN).float()

    # Calculate IoU
    iou = TP.float() / (TP + FP + FN).float()

    return dice, iou
    
def post_process_mask(mask):
    # Apply morphological closing and then opening
    struct_elem = np.ones((3, 3, 3))  # Adjust the structure element size as needed
    mask = scipy.ndimage.binary_closing(mask, structure=struct_elem).astype(np.int32)
    mask = scipy.ndimage.binary_opening(mask, structure=struct_elem).astype(np.int32)
    return mask


# Function to save checkpoint
def save_checkpoint(state, filename="unet_checkpoint_comb_weighted_11+1.pth"):
    torch.save(state, filename)

# Function to load checkpoint
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

def calculate_metrics(y_true, y_pred):
    # Flatten the tensors and convert them to numpy arrays if they aren't already
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Ensure y_true is an integer array (binary format)
    y_true = y_true.astype(np.int32)

    # Convert probabilities in y_pred to binary format
    y_pred_binary = (y_pred > 0.5).astype(np.int32)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, average='binary')
    recall = recall_score(y_true, y_pred_binary, average='binary')
    f1 = f1_score(y_true, y_pred_binary, average='binary')
    return accuracy, precision, recall, f1



    
def get_mean_image_size(image_paths):
    total_size = np.array([0, 0, 0])
    num_images = len(image_paths)

    for path in image_paths:
        image = nib.load(path).get_fdata()
        total_size += np.array(image.shape)

    mean_size = total_size / num_images
    return mean_size
    

    

data_folder_path = '/path/to/image_dataset'

# Initialize the full dataset for splitting
full_dataset = NiftiDataset(data_folder_path, is_train=True)  # is_train flag here is not crucial

# Calculate the total number of samples
total_size = len(full_dataset)

# Calculate train, validation, and test sizes
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

# Split the dataset
train_indices, val_indices, test_indices = random_split(range(total_size), [train_size, val_size, test_size])

# Create dataset instances for each split with appropriate is_train flag
train_dataset = NiftiDataset(data_folder_path, is_train=True, indices=train_indices)
val_dataset = NiftiDataset(data_folder_path, is_train=False, indices=val_indices)
test_dataset = NiftiDataset(data_folder_path, is_train=False, indices=test_indices)

# Print file names for each dataset
print("Training Dataset Files:")
train_dataset.print_file_names()

print("\nValidation Dataset Files:")
val_dataset.print_file_names()

print("\nTest Dataset Files:")
test_dataset.print_file_names()


# DataLoaders for each split
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Print file names for each batch in the DataLoader
print("Training DataLoader Files:")
print_dataloader_file_names(train_dataloader, train_dataset)

print("\nValidation DataLoader Files:")
print_dataloader_file_names(val_dataloader, val_dataset)

print("\nTest DataLoader Files:")
print_dataloader_file_names(test_dataloader, test_dataset)




# Output directory
output_dir = '/path/to/output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)





# Initialize model, loss function, and optimizer
device = torch.device("cpu")
model = UNet3D().to(device)
criterion = CombinedDiceBCELoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-5)  # Adjust weight_decay as needed

is_cuda_available = torch.cuda.is_available()
print("Is CUDA (GPU support) available:", is_cuda_available)

print("Model is on:", device)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# TensorBoard setup
writer = SummaryWriter('runs/combined weighted loss_11+1')


    
# Early stopping parameters
patience = 10  # Number of epochs to wait after last time validation loss improved
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_checkpoint = None
start_epoch = 0
early_stopping_triggered = False

# Define checkpoint paths
regular_checkpoint_file = "unet_checkpoint_comb_weighted_11+1.pth"
best_model_checkpoint_file = "unet3d_best_model_comb_weighted_checkpoint_11+1.pth"

# Check if the best model checkpoint exists
if os.path.isfile(best_model_checkpoint_file):
    print("Loading best model checkpoint.")
    checkpoint = torch.load(best_model_checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
# If not, check if the regular checkpoint exists and load it
elif os.path.isfile(regular_checkpoint_file):
    print("Loading regular checkpoint.")
    checkpoint = torch.load(regular_checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint.get('epoch', 0)
else:
    print("No checkpoint found. Starting training from scratch.")

# Training loop
num_epochs = 80

for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    running_dice_loss = 0.0  # Initialize running dice loss
    running_bce_loss = 0.0   # Initialize running BCE loss
    running_dice_score = 0.0  # Initialize running dice score
    running_iou_score = 0.0   # Initialize running IoU score
    
    for i, batch in enumerate(train_dataloader):
        inputs = batch['image'].to(device).float()
        labels = batch['mask'].to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        combined_loss, dice_loss, bce_loss = criterion(outputs, labels)
        combined_loss.backward()
        optimizer.step()

        running_loss += combined_loss.item()
        running_dice_loss += dice_loss.item()
        running_bce_loss += bce_loss.item()
        dice_score, iou_score = calculate_dice_iou(torch.sigmoid(outputs)>0.5, labels>0.5)
        running_dice_score += dice_score.item()
        running_iou_score += iou_score.item()


        # Log individual components for each batch
        writer.add_scalar('Training Dice Loss', dice_loss.item(), epoch * len(train_dataloader) + i)
        writer.add_scalar('Training BCE Loss', bce_loss.item(), epoch * len(train_dataloader) + i)   
        writer.add_scalar('Training Dice Score', dice_score.item(), epoch * len(train_dataloader) + i)
        writer.add_scalar('Training IoU Score', iou_score.item(), epoch * len(train_dataloader) + i)   
        # Log training loss for each iteration
        writer.add_scalar('Training Loss', combined_loss.item(), epoch * len(train_dataloader) + i)
        

        # Log a sample prediction every few batches
        if i % 5 == 0:  # Adjust as needed
            # Select a single slice for visualization
            slice_index = outputs.shape[4] // 2  # Middle slice
            inputs_slice = inputs[:, :, :, :, slice_index]
            outputs_slice = torch.sigmoid(outputs[:, :, :, :, slice_index])
            labels_slice = labels[:, :, :, :, slice_index]
            
            writer.add_images('Inputs', inputs_slice, epoch * len(train_dataloader) + i, dataformats='NCHW')
            writer.add_images('Predictions', outputs_slice, epoch * len(train_dataloader) + i, dataformats='NCHW')
            writer.add_images('Ground Truth', labels_slice, epoch * len(train_dataloader) + i, dataformats='NCHW')
            
    # Log average loss components for the epoch
    avg_dice_loss = running_dice_loss / len(train_dataloader)
    avg_bce_loss = running_bce_loss / len(train_dataloader)
    epoch_loss = running_loss / len(train_dataloader)
    avg_dice_score = running_dice_score / len(train_dataloader)
    avg_iou_score = running_iou_score / len(train_dataloader)
    writer.add_scalar('Average Training Dice Loss', avg_dice_loss, epoch)
    writer.add_scalar('Average Training BCE Loss', avg_bce_loss, epoch)
    writer.add_scalar('Average Training epoch Loss', epoch_loss, epoch)
    writer.add_scalar('Average Training Dice Score', avg_dice_score, epoch)
    writer.add_scalar('Average Training IoU Score', avg_iou_score, epoch)
    
    
     # Validation phase
    model.eval()
    val_loss = 0.0
    val_dice_loss = 0.0  # Initialize running dice loss
    val_bce_loss = 0.0   # Initialize running BCE loss
    val_dice_score = 0.0  # Initialize running dice score
    val_iou_score = 0.0   # Initialize running IoU score
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            inputs = batch['image'].to(device).float()
            labels = batch['mask'].to(device).float()
            
            outputs = model(inputs)
            combined_loss, dice_loss, bce_loss = criterion(outputs, labels)
            val_loss += combined_loss.item()
            val_dice_loss += dice_loss.item()
            val_bce_loss += bce_loss.item()
            dice_score, iou_score = calculate_dice_iou(torch.sigmoid(outputs)>0.5, labels>0.5)
            val_dice_score += dice_score.item()
            val_iou_score += iou_score.item()

            # Log individual components for each batch
            writer.add_scalar('Validation Dice Loss', dice_loss.item(), epoch * len(val_dataloader) + i)
            writer.add_scalar('Validation BCE Loss', bce_loss.item(), epoch * len(val_dataloader) + i) 
            writer.add_scalar('Validation Dice Score', dice_score.item(), epoch * len(val_dataloader) + i)
            writer.add_scalar('Validation IoU Score', iou_score.item(), epoch * len(val_dataloader) + i)     
            # Log training loss for each iteration
            writer.add_scalar('Validation Loss', combined_loss.item(), epoch * len(train_dataloader) + i)
            
            # Additional logging to TensorBoard if needed

    val_loss /= len(val_dataloader)
    avg_val_dice_loss = val_dice_loss / len(val_dataloader)
    avg_val_bce_loss = val_bce_loss / len(val_dataloader)
    avg_val_dice_score = val_dice_score / len(val_dataloader)
    avg_val_iou_score = val_iou_score / len(val_dataloader)
    writer.add_scalar('Average Validation Dice Loss', avg_val_dice_loss, epoch)
    writer.add_scalar('Average Validation BCE Loss', avg_val_bce_loss, epoch)
    writer.add_scalar('Average Validation Dice Score', avg_val_dice_score, epoch)
    writer.add_scalar('Average Validation IoU Score', avg_val_iou_score, epoch)
    writer.add_scalar('Average Validation epoch Loss', val_loss, epoch)
    
    
    
    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # Save best model checkpoint immediately when found
        best_model_checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss
        }
        
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            early_stopping_triggered = True
            break  # Break out of the loop

    # Update learning rate
    scheduler.step(epoch_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, "
      f"Loss: {epoch_loss}, "
      f"Dice Loss: {avg_dice_loss}, "
      f"BCE Loss: {avg_bce_loss}, "
      f"Train Dice: {avg_dice_score}, "  # Added
      f"Train IoU: {avg_iou_score}, "    # Added
      f"Val Loss: {val_loss}, "
      f"Val Dice Loss: {avg_val_dice_loss}, "
      f"Val BCE Loss: {avg_val_bce_loss}, "
      f"Val Dice: {avg_val_dice_score}, "      # Added
      f"Val IoU: {avg_val_iou_score}")          # Added

    # Save checkpoint at the end of each epoch
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    save_checkpoint(checkpoint)

# Save the best model checkpoint after the training loop
if best_model_checkpoint is not None:
    torch.save(best_model_checkpoint, 'unet3d_best_model_comb_weighted_checkpoint_11+1.pth')
    print("Best model checkpoint saved")    
    
    
# If early stopping was triggered, load the best model to continue
if early_stopping_triggered and best_model_checkpoint:
    model.load_state_dict(best_model_checkpoint['state_dict'])
    
# Test Loop with TensorBoard Visualization
model.eval()
test_loss = 0.0
test_dice_loss = 0.0  # Initialize running dice loss
test_bce_loss = 0.0   # Initialize running BCE loss

# Initialize lists for metrics
all_accuracies = []
all_precisions = []
all_recalls = []
all_f1s = []
all_dice = []
all_IoU = []

with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        inputs = batch['image'].to(device).float()  # Ensure inputs are float
        labels = batch['mask'].to(device).float()   # Ensure labels are float
        outputs = model(inputs)
        combined_loss, dice_loss, bce_loss = criterion(outputs, labels)
        test_loss += combined_loss.item()
        test_dice_loss += dice_loss.item()
        test_bce_loss += bce_loss.item()
        
        # Apply post-processing to the output mask
        output_mask_np = torch.sigmoid(outputs).detach().cpu().squeeze().numpy() > 0.5
        #output_mask_np = post_process_mask(output_mask_np)

        
        # Calculate and store metrics
        acc, prec, rec, f1 = calculate_metrics(labels.cpu().numpy()>0.5, output_mask_np)
        # After you have the outputs from your model
        dice_score, iou_score = calculate_dice_iou(torch.sigmoid(outputs).cpu()>0.5, labels.cpu()>0.5)
        all_accuracies.append(acc)
        all_precisions.append(prec)
        all_recalls.append(rec)
        all_f1s.append(f1)
        all_dice.append(dice_score)
        all_IoU.append(iou_score)

        # Select a single slice for visualization
        slice_index = outputs.shape[4] // 2  # Middle slice
        outputs_slice = outputs[:, :, :, :, slice_index]
        labels_slice = labels[:, :, :, :, slice_index]

        writer.add_images('Test/Input', inputs[:, :, :, :, slice_index], i, dataformats='NCHW')
        writer.add_images('Test/Label', labels_slice, i, dataformats='NCHW')
        writer.add_images('Test/Prediction', torch.sigmoid(outputs_slice), i, dataformats='NCHW')
        
        # Save the output masks
        

        # Convert boolean mask to an integer type
        output_mask = output_mask_np.astype(np.int8)
        output_nifti = nib.Nifti1Image(output_mask, affine=np.eye(4))
        nib.save(output_nifti, os.path.join(output_dir, f'prediction_batch{i+1}.nii'))
        
# Average metrics over all test batches
avg_accuracy = np.mean(all_accuracies)
avg_precision = np.mean(all_precisions)
avg_recall = np.mean(all_recalls)
avg_f1 = np.mean(all_f1s)
avg_dice = np.mean(all_dice)
avg_IoU = np.mean(all_IoU)


# Log average metrics to TensorBoard
writer.add_scalar('Test/Average Accuracy', avg_accuracy)
writer.add_scalar('Test/Average Precision', avg_precision)
writer.add_scalar('Test/Average Recall', avg_recall)
writer.add_scalar('Test/Average F1-Score', avg_f1)
writer.add_scalar('Test/Average Dice-Score', avg_dice)
writer.add_scalar('Test/Average IoU-Score', avg_IoU)

print(f"Test Metrics: Accuracy: {avg_accuracy}, Precision: {avg_precision}, Recall: {avg_recall}, F1-Score: {avg_f1}, Dice-Score: {avg_dice}, IoU-Score: {avg_IoU}")

test_loss /= len(test_dataloader)
test_dice_loss /= len(test_dataloader)
test_bce_loss /= len(test_dataloader)
print(f'Test Loss: {test_loss}, Test Dice Loss: {test_dice_loss}, Test BCE Loss: {test_bce_loss}')


# Close the TensorBoard writer
writer.close()

# Notify when the training is complete
print("Training complete!")

# Save the trained model
torch.save(model.state_dict(), 'unet3d_model_comb_weighted_11+1.pth')
print("Model saved")