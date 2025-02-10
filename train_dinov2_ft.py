import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from timm.data.auto_augment import rand_augment_transform
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import v2 as T

# ===========================
# CONFIGURATION
# ===========================
BATCH_SIZE = 8
EPOCHS = 100  # Increased for better fine-tuning
LEARNING_RATE = 1e-4
AUGMENT_FACTOR = 10  # More augmented versions per image
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================
# DATA AUGMENTATION
# ===========================
def get_transforms(is_train):
    """Applies RandAugment along with resizing and normalization."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((448, 448)),  # Resize all images to 448x448
            rand_augment_transform(config_str="rand-m9-mstd0.5", hparams={}),  # RandAugment
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((448, 448)),  # Resize all images to 448x448
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize
        ])

def tensor_to_pil(image_tensor):
    """Convert a normalized tensor image to a PIL image."""
    if image_tensor.min() < 0:  # If in range [-1,1], rescale to [0,1]
        image_tensor = (image_tensor + 1) / 2

    image_tensor = torch.clamp(image_tensor, 0, 1)  # Ensure values are in [0,1]

    to_pil = transforms.ToPILImage()
    return to_pil(image_tensor)
        
# ===========================
# CUSTOM DATASET CLASS
# ===========================
class AugmentedBirdDataset(Dataset):
    def __init__(self, metadata_csv, root_dir, augment_factor, is_train=True):
        self.data = pd.read_csv(metadata_csv)
        self.root_dir = root_dir
        self.is_train = is_train
        self.augment_factor = augment_factor
        self.transform = get_transforms(self.is_train)
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

    def __len__(self):
        return len(self.data) * self.augment_factor  # Expanded dataset size

    def __getitem__(self, idx):
        """Loads an image, applies augmentation, and processes it for the model."""
        original_idx = idx // self.augment_factor
        img_path = os.path.join(self.root_dir, self.data.iloc[original_idx, 0])
        label = self.data.iloc[original_idx, 2]  # Assuming label is in 3rd column

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)  # Apply RandAugment
        img = tensor_to_pil(img)
        inputs = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        return inputs, torch.tensor(label, dtype=torch.long)

# ===========================
# MODEL SETUP
# ===========================
class FineTunedDINOv2(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedDINOv2, self).__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-large")
        
        # Freeze all layers except the last 1 transformer layer & MLP head
        for param in self.backbone.parameters():
            param.requires_grad = False

        for layer in self.backbone.encoder.layer[-1:]:  # Unfreeze 1 transformer layers
            for param in layer.parameters():
                param.requires_grad = True
        
         # Define a deeper classification head with GELU and Dropout
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, num_classes)  # Final classification layer
        )

    def forward(self, x):
        x = self.backbone(x).last_hidden_state[:, 0, :]  # Extract CLS token
        x = self.classifier(x)  # Pass through new classification head
        return x

# ===========================
# TRAINING FUNCTION
# ===========================
def train(model, train_loader, val_loader, epochs, optimizer, criterion, scheduler):
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss, train_correct, total = 0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * train_correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc)
        
        # Validation Phase
        model.eval()
        val_loss, val_correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = 100 * val_correct / total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"Saving best model with accuracy: {best_acc:.2f}%")
            torch.save(model.state_dict(), "best_dinov2_ft_model.pth")

        # Learning Rate Scheduling
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%\n")

    # Plot and Save Results
    plot_results(train_losses, val_losses, train_accuracies, val_accuracies)
    evaluate_model(model, train_loader, "Train")
    evaluate_model(model, val_loader, "Validation")

# ===========================
# PLOTTING FUNCTION
# ===========================
def plot_results(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Over Epochs")
    plt.savefig("loss_plot.png")

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.legend()
    plt.title("Accuracy Over Epochs")
    plt.savefig("accuracy_plot.png")

# ===========================
# EVALUATION FUNCTION
# ===========================
def evaluate_model(model, data_loader, dataset_name):
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.savefig(f"{dataset_name.lower()}_confusion_matrix.png")

    # Classification Report
    print(f"\n{dataset_name} Classification Report:\n", classification_report(all_labels, all_preds))
    
# ===========================
# TESTING FUNCTION
# ===========================
def generate_test_predictions(model, test_folder):
    model.load_state_dict(torch.load("best_dinov2_ft_model.pth"))
    model.eval()
    predictions = []

    for img_name in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_name)
        img = Image.open(img_path)
        img = img.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img)
            pred_label = output.argmax(dim=1).item()

        predictions.append({"path": img_name, "class_idx": pred_label})

    df = pd.DataFrame(predictions)
    df.to_csv("dinov2_ft_submission.csv", index=False)
    print("Test predictions saved to dinov2_ft_submission.csv")

# ===========================
# MAIN FUNCTION
# ===========================
if __name__ == "__main__":
    #paths
    root_train = "bdma-07-competition/BDMA7_project_files/train_images"
    root_val = "bdma-07-competition/BDMA7_project_files/val_images"
    train_csv = "bdma-07-competition/BDMA7_project_files/train_metadata.csv"
    val_csv = "bdma-07-competition/BDMA7_project_files/val_metadata.csv"
    test_folder = "bdma-07-competition/BDMA7_project_files/test_images/mistery_cat"
    
    # Load Dataset
    train_dataset = AugmentedBirdDataset(train_csv, root_train, AUGMENT_FACTOR, True)
    val_dataset = AugmentedBirdDataset(val_csv, root_val, 1, False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    num_classes = 20
    model = FineTunedDINOv2(num_classes).to(DEVICE)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, val_loader, EPOCHS, optimizer, criterion, scheduler)
    generate_test_predictions(model, test_folder)