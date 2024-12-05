"""
#1
"""
!pip install ultralytics
!pip install torch torchvision
!pip install albumentations
!pip install timm
!pip install wandb
!pip install seaborn # Install the seaborn library
"""
END OF #1
"""



"""
#2 (U can import your own dataset)
""""
import kagglehub

# Download latest version
path = kagglehub.dataset_download("kritikseth/fruit-and-vegetable-image-recognition")

print("Path to dataset files:", path)
"""
END OF #2
"""


"""
#3 Edited my dataset (optional)
"""
import shutil

# List of directories to delete
directories = [
    "/root/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/8/train/beetroot",
    "/root/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/8/train/cabbage",
    "/root/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/8/train/carrot",
    "/root/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/8/train/cauliflower",
    "/root/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/8/train/cucumber",
    "/root/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/8/train/garlic",
    "/root/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/8/train/ginger",
    "/root/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/8/train/lettuce",
    "/root/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/8/train/peas",
    "/root/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/8/train/potato",
    "/root/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/8/train/raddish",
    "/root/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/8/train/soy beans",
    "/root/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/8/train/spinach",
    "/root/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/8/train/sweetpotato",
    "/root/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/8/train/turnip"
]

# Delete each directory and its contents
for directory in directories:
    shutil.rmtree(directory)
    print(f"Directory {directory} has been deleted.")
"""
END OF #3
"""


"""
#4 cuda
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import random

`
# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create folder for saving results
os.makedirs('results', exist_ok=True)
"""
END OF #4
"""


"""
#5 Exporting dataset structure/classes
"""
def explore_data(data_path):
    """Explore and visualize the dataset"""
    print("\nExploring Dataset Structure:")
    print("-" * 50)

    splits = ['train', 'validation', 'test']
    for split in splits:
        split_path = os.path.join(data_path, split)
        if os.path.exists(split_path):
            classes = sorted(os.listdir(split_path))
            total_images = sum(len(os.listdir(os.path.join(split_path, cls)))
                               for cls in classes)

            print(f"\n{split.capitalize()} Set:")
            print(f"Number of classes: {len(classes)}")
            print(f"Total images: {total_images}")
            print(f"Example classes: {', '.join(classes[:5])}...")

    # Visualize sample images
    print("\nVisualizing Sample Images...")
    train_path = os.path.join(data_path, 'train')

    # Check if the train folder exists
    if os.path.exists(train_path):
        classes = sorted(os.listdir(train_path))

        plt.figure(figsize=(15, 10))
        for i in range(9):
            class_name = random.choice(classes)
            class_path = os.path.join(train_path, class_name)

            # Get the list of files in the directory
            files = os.listdir(class_path)

            # Check if directory is empty
            if files:
                img_name = random.choice(files)
                img_path = os.path.join(class_path, img_name)

                img = Image.open(img_path)
                plt.subplot(3, 3, i + 1)
                plt.imshow(img)
                plt.title(f'Class: {class_name}')
                plt.axis('off')
            else:
                print(f"Skipping empty directory: {class_path}")

        plt.tight_layout()
        plt.savefig('results/sample_images.png')
        plt.show()
    else:
        print(f"Train folder not found at: {train_path}")


# Explore dataset
data_path = "/root/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/8"  # can be change depending on the path of your data set
explore_data(data_path)
"""
END OF #5
"""


"""
#6 Augmentations
"""
class FruitVegDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.images = []
        self.labels = []

        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_path, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Visualize augmentations
def show_augmentations(dataset, num_augments=5):
    """Show original image and its augmented versions"""
    idx = random.randint(0, len(dataset)-1)
    img_path = dataset.images[idx]
    original_img = Image.open(img_path).convert('RGB')

    plt.figure(figsize=(15, 5))

    # Show original
    plt.subplot(1, num_augments+1, 1)
    plt.imshow(original_img)
    plt.title('Original')
    plt.axis('off')

    # Show augmented versions
    for i in range(num_augments):
        augmented = train_transform(original_img)
        augmented = augmented.permute(1, 2, 0).numpy()
        augmented = (augmented * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])
        augmented = np.clip(augmented, 0, 1)

        plt.subplot(1, num_augments+1, i+2)
        plt.imshow(augmented)
        plt.title(f'Augmented {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/augmentations.png')
    plt.show()

# Create datasets and show augmentations
train_dataset = FruitVegDataset(data_path, 'train', train_transform)
show_augmentations(train_dataset)
"""
END OF #6
"""


"""
#7 Feature Maps
"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv(x)

class FruitVegCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 512)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Function to visualize feature maps
def visualize_feature_maps(model, sample_image):
    """Visualize feature maps after each conv block"""
    model.eval()

    # Get feature maps after each conv block
    feature_maps = []
    x = sample_image.unsqueeze(0).to(device)

    for block in model.features:
        x = block(x)
        feature_maps.append(x.detach().cpu())

    # Plot feature maps
    plt.figure(figsize=(15, 10))
    for i, fmap in enumerate(feature_maps):
        # Plot first 6 channels of each block
        fmap = fmap[0][:6].permute(1, 2, 0)
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())

        for j in range(min(6, fmap.shape[-1])):
            plt.subplot(5, 6, i*6 + j + 1)
            plt.imshow(fmap[:, :, j], cmap='viridis')
            plt.title(f'Block {i+1}, Ch {j+1}')
            plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/feature_maps.png')
    plt.show()

# Initialize model and visualize feature maps
model = FruitVegCNN(num_classes=len(train_dataset.classes)).to(device)
sample_image, _ = train_dataset[0]
visualize_feature_maps(model, sample_image)
"""
END OF #7
"""


"""
#8 optimized_model.pth (Epoch over 30)
"""
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(val_loader), 100. * correct / total

def plot_training_progress(history):
    """Plot and save training progress"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/training_progress.png')
    plt.show()

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_dataset = FruitVegDataset(data_path, 'validation', val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training loop
num_epochs = 30
best_val_acc = 0
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

print("\nStarting training...")
for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device)

    val_loss, val_acc = validate(
        model, val_loader, criterion, device)

    # Update scheduler
    scheduler.step(val_loss)

    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

    # Plot progress
    if (epoch + 1) % 5 == 0:
        plot_training_progress(history)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f'New best validation accuracy: {best_val_acc:.2f}%')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_val_acc,
        }, 'results/best_model.pth')

# Final training visualization
plot_training_progress(history)
"""
END OF #8
"""


"""
#9 Plot the optimized_model.pth 
"""
import seaborn as sns # Import the seaborn library
import matplotlib.pyplot as plt

def plot_optimized_results(history):
    # Register Seaborn styles with Matplotlib
    sns.set()  # Apply default Seaborn style

    plt.figure(figsize=(15, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training', marker='o')
    plt.plot(history['val_acc'], label='Validation', marker='o')
    plt.title('Model Accuracy with Optimizations')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training', marker='o')
    plt.plot(history['val_loss'], label='Validation', marker='o')
    plt.title('Model Loss with Optimizations')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('optimized_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print best metrics
    best_train_acc = max(history['train_acc'])
    best_val_acc = max(history['val_acc'])
    print(f"\nBest Training Accuracy: {best_train_acc:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

# Plot results
plot_optimized_results(history)
"""
END OF #9
"""


"""
#10 Train the real model Entire & State (Epoch 1/50)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler


# Improved training configurations
class OptimizedConfig:
    def __init__(self):
        self.image_size = 256  # Increased from 224
        self.batch_size = 16  # Smaller batch size for better generalization
        self.learning_rate = 3e-4
        self.weight_decay = 0.01
        self.epochs = 50
        self.dropout = 0.3


# Enhanced data augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Optimized model architecture
class OptimizedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Use pretrained ResNet50 as backbone
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0',
                                       'resnet50', pretrained=True)

        # Freeze early layers
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False

        # Modified classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# Optimized training function
def train_with_optimization(model, train_loader, val_loader, config):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay)

    # One Cycle Learning Rate Scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )

    # Gradient Scaler for mixed precision training
    scaler = GradScaler()

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })

        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100. * correct / total
        val_loss = val_loss / len(val_loader)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'\nEpoch {epoch + 1}/{config.epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_val_acc,
            }, 'optimized_model.pth')
            print(f'New best validation accuracy: {best_val_acc:.2f}%')

    return history


# Create dataloaders with optimized configuration
config = OptimizedConfig()
train_dataset = FruitVegDataset(data_path, 'train', train_transform)
val_dataset = FruitVegDataset(data_path, 'validation', val_transform)

train_loader = DataLoader(train_dataset,
                          batch_size=config.batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)
val_loader = DataLoader(val_dataset,
                        batch_size=config.batch_size,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True)

# Initialize and train optimized model
model = OptimizedCNN(num_classes=len(train_dataset.classes)).to(device)
history = train_with_optimization(model, train_loader, val_loader, config)
"""
END OF #10
"""


"""
#11 Plot the trained model Entire/State
"""


def plot_optimized_results(history):
    import seaborn as sns
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(15, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training', marker='o')
    plt.plot(history['val_acc'], label='Validation', marker='o')
    plt.title('Model Accuracy with Optimizations')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training', marker='o')
    plt.plot(history['val_loss'], label='Validation', marker='o')
    plt.title('Model Loss with Optimizations')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('optimized_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print best metrics
    best_train_acc = max(history['train_acc'])
    best_val_acc = max(history['val_acc'])
    print(f"\nBest Training Accuracy: {best_train_acc:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")


# Plot results
plot_optimized_results(history)
"""
END OF #11
"""


"""
#12 Predicting an Image that has never seen by the model
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO


# Load the saved model
def load_model():
    # Check if model file exists
    try:
        # Load model checkpoint
        checkpoint = torch.load('optimized_model.pth')

        # Instantiate model with the correct number of classes
        # The original model was trained with 23 classes, not 36.
        model = OptimizedCNN(num_classes=23)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully!")
        return model
    except FileNotFoundError:
        print("Model file 'optimized_model.pth' not found!")
        return None


# Prediction function
def predict_image(url, model):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load image from URL
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')

    # Transform image
    input_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get top 5 predictions
        top_probs, top_indices = torch.topk(probabilities, 5)

    # Show results
    plt.figure(figsize=(12, 4))

    # Show image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')

    # Show predictions
    plt.subplot(1, 2, 2)
    classes = sorted(
        os.listdir("/root/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/8/train"))
    y_pos = range(5)
    plt.barh(y_pos, [prob.item() * 100 for prob in top_probs])
    plt.yticks(y_pos, [classes[idx] for idx in top_indices])
    plt.xlabel('Probability (%)')
    plt.title('Top 5 Predictions')

    plt.tight_layout()
    plt.show()

    # Print predictions
    print("\nPredictions:")
    print("-" * 30)
    for i in range(5):
        print(f"{classes[top_indices[i]]:20s}: {top_probs[i] * 100:.2f}%")


# Load model
model = load_model()

# Now you can use it like this:
predict_image('https://extension.umn.edu/sites/extension.umn.edu/files/misshapen-strawberries-aklodd.jpg', model)
"""
END OF #12
"""


"""
#13 Saving the model
"""
torch.save(model, 'Fruit_Recognition_Entire_Model.pth')
torch.save(model.state_dict(), 'Fruit_Recognition_State_Model.pth')
"""
END OF #13
"""


"""
#14 Downloading the entire content folder
"""
# download the content directory
from google.colab import files
import os
import shutil

# Create a 'content' directory if it doesn't exist
if not os.path.exists('content'):
    os.makedirs('content')
else:
    # If 'content' directory exists, remove it and its contents
    shutil.rmtree('content')
    os.makedirs('content')  # Create it again

# Copy the files you want to download to the 'content' directory
shutil.copytree('results', 'content/results')
# Use shutil.copy to copy single files
shutil.copy('optimized_results.png', 'content/optimized_results.png')
shutil.copy('optimized_model.pth', 'content/optimized_model.pth')
shutil.copy('Fruit_Recognition_Entire_Model.pth', 'content/Fruit_Recognition_Entire_Model.pth')
shutil.copy('Fruit_Recognition_State_Model.pth', 'content/Fruit_Recognition_State_Model.pth')

# Create the 'content.zip' file
shutil.make_archive('content', 'zip', 'content')

# Now, download the 'content.zip' file
files.download('content.zip')
"""
END OF #14
"""
