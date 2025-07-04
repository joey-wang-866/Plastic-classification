import os
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img
from torchvision import datasets
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

import torch_testing

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

learning_rate = 1e-4
batch_size = 4
num_epochs = 25

class SchoolMealDataset(Dataset):
    def __init__(self, root, split="train", transform=None, test_split=0.2, seed=42):
        self.root = root
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = []

        random.seed(seed)
        for label_idx, label_name in enumerate(sorted(os.listdir(root))):
            label_dir = os.path.join(root, label_name)
            if os.path.isdir(label_dir):
                images = [os.path.join(label_dir, img_name) for img_name in os.listdir(label_dir)]
                # images = images[:int(len(images) * 0.005)]
                random.shuffle(images)
                split_idx = int(len(images) * (1 - test_split))
                if split == "train":
                    selected_images = images[:split_idx]
                elif split == "test":
                    selected_images = images[split_idx:]
                else:
                    raise ValueError("Invalid split name. Use 'train' or 'test'.")

                self.image_paths.extend(selected_images)
                self.labels.extend([label_idx] * len(selected_images))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        label = int(label)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
train_dataset = SchoolMealDataset(root='./Garbage_classification', split="train", transform=transform)
test_dataset = SchoolMealDataset(root='./Garbage_classification', split="test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
for images, labels in train_loader:
    image = images[1].permute(1, 2, 0).cpu().numpy()
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    break
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

class VGGModel_16(nn.Module):
    def __init__(self):
        super(VGGModel_16, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        for param in self.vgg.parameters():
            param.requires_grad = True
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, 50)

    def forward(self, x):
        return self.vgg(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_16 = VGGModel_16().to(device)
criterion_16 = nn.CrossEntropyLoss()
optimizer_16 = optim.Adam(model_16.parameters(), lr=learning_rate)
scaler_16 = GradScaler()
print(summary(model_16, (3, 224, 224)))

class ResNetModel_50(nn.Module):
    def __init__(self):
        super(ResNetModel_50, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = True
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 50)

    def forward(self, x):
        return self.resnet(x)


model_50 = ResNetModel_50().to(device)
criterion_50 = nn.CrossEntropyLoss()
optimizer_50 = optim.Adam(model_50.parameters(), lr=learning_rate)
scaler_50 = GradScaler()
print(summary(model_50, (3, 224, 224)))

dataset_size = len(train_loader.dataset)
subset_size = int(0.01 * dataset_size)
indices = np.random.permutation(dataset_size)[:subset_size]
subset_dataset = Subset(train_loader.dataset, indices)
train_loader_evaluate = torch.utils.data.DataLoader(
    subset_dataset,
    batch_size=train_loader.batch_size,
    shuffle=True,
    num_workers=train_loader.num_workers,
    pin_memory=train_loader.pin_memory
)
dataset_size_evaluate = len(train_loader_evaluate.dataset)
print(f"Original dataset size: {dataset_size}")
print(f"Dataset size for evaluation: {dataset_size_evaluate}")

def evaluate_on_dataset(model, criterion, loader):
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            # Calculate Top-1 accuracy
            _, predicted_top1 = torch.max(outputs.data, 1)
            correct_top1 += (predicted_top1 == labels).sum().item()

            # Store labels and predictions for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted_top1.cpu().numpy())

            # Calculate Top-5 accuracy
            _, predicted_top5 = torch.topk(outputs.data, 5, dim=1)
            correct_top5 += torch.eq(predicted_top5, labels.view(-1, 1)).sum().item()

            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy_top1 = 100 * correct_top1 / total
    accuracy_top5 = 100 * correct_top5 / total

    return avg_loss, accuracy_top1, accuracy_top5, all_labels, all_preds

def train_model(model, criterion, optimizer, scaler, num_epochs, name):
    train_losses = []
    train_accuracies_top1 = []
    train_accuracies_top5 = []
    test_losses = []
    test_accuracies_top1 = []
    test_accuracies_top5 = []
    all_labels = []
    all_preds = []
    best_test_accuracy_top1 = 0.0

    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        print(f'Starting epoch {epoch + 1}/{num_epochs}')
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # scheduler.step()
                total_loss += loss.item() * labels.size(0)
                pbar.update(1)

        avg_train_loss = total_loss / len(train_loader.dataset)

        train_loss, train_accuracy_top1, train_accuracy_top5, epoch_labels, epoch_preds = evaluate_on_dataset(model, criterion, train_loader_evaluate)
        train_losses.append(train_loss)
        train_accuracies_top1.append(train_accuracy_top1)
        train_accuracies_top5.append(train_accuracy_top5)
        
        test_loss, test_accuracy_top1, test_accuracy_top5, epoch_labels, epoch_preds = evaluate_on_dataset(model, criterion, test_loader)
        test_losses.append(test_loss)
        test_accuracies_top1.append(test_accuracy_top1)
        test_accuracies_top5.append(test_accuracy_top5)
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Training Loss: {train_loss:.2f}, Training Accuracy (top 1): {train_accuracy_top1:.2f}%, , Training Accuracy (top 5): {train_accuracy_top5:.2f}%')
        print(f'Test Loss: {test_loss:.2f}, Test Accuracy (top 1): {test_accuracy_top1:.2f}%, Test Accuracy (top 5): {test_accuracy_top5:.2f}%')
        all_labels = epoch_labels
        all_preds = epoch_preds

        # Early stopping 判斷
        if len(test_losses) > 5 and test_losses[-1] > test_losses[-2] > test_losses[-3] > test_losses[-4] > test_losses[-5]:
            print("Early stopping triggered")
            break

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", mask=(cm == 0))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {name}")
    plt.show()
    print(f"Model {name} training finish!")

    return train_losses, train_accuracies_top1, train_accuracies_top5, test_losses, test_accuracies_top1, test_accuracies_top5

start_time_VGG16 = datetime.datetime.now()
train_losses_16, train_accuracies_16_top1, train_accuracies_16_top5, test_losses_16, test_accuracies_16_top1, test_accuracies_16_top5 = train_model(model_16, criterion_16, optimizer_16, scaler_16, num_epochs=num_epochs, name="VGG16")
end_time_VGG16 = datetime.datetime.now()
elapsed_time = end_time_VGG16 - start_time_VGG16

hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
minutes, seconds = divmod(remainder, 60)

elapsed_time_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
print(f"VGG16 training time: {elapsed_time_str}")

start_time_Resnet50 = datetime.datetime.now()
train_losses_50, train_accuracies_50_top1, train_accuracies_50_top5, test_losses_50, test_accuracies_50_top1, test_accuracies_50_top5 = train_model(model_50, criterion_50, optimizer_50, scaler_50, num_epochs=num_epochs, name="Resnet50")
end_time_Resnet50 = datetime.datetime.now()
elapsed_time = end_time_Resnet50 - start_time_Resnet50

hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
minutes, seconds = divmod(remainder, 60)

elapsed_time_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
print(f"Resnet50 training time: {elapsed_time_str}")

plt.figure(figsize=(10, 15))
plt.subplot(4, 1, 1)
plt.plot(train_losses_50, label='Training Loss')
plt.plot(test_losses_50, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Resnet50 Training and Validation Loss')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(train_losses_16, label='Training Loss')
plt.plot(test_losses_16, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('VGG16 Training and Validation Loss')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(train_accuracies_50_top1, label='Top 1 Training Accuracy')
plt.plot(train_accuracies_50_top5, label='Top 5 Training Accuracy')
plt.plot(test_accuracies_50_top1, label='Top 1 Validation Accuracy')
plt.plot(test_accuracies_50_top5, label='Top 5 Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Resnet50 Training and Validation Accuracy')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(train_accuracies_16_top1, label='Top 1 Training Accuracy')
plt.plot(train_accuracies_16_top5, label='Top 5 Training Accuracy')
plt.plot(test_accuracies_16_top1, label='Top 1 Validation Accuracy')
plt.plot(test_accuracies_16_top5, label='Top 5 Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('VGG16 Training and Validation Accuracy')

plt.tight_layout()
plt.legend()
plt.show()

plt.figure(figsize=(10, 15))
plt.subplot(4, 1, 1)
plt.plot(train_accuracies_16_top1, label='VGG16')
plt.plot(train_accuracies_50_top1, label='Resnet50')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Resnet50 vs VGG16 Top 1 Training Accuracy')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(train_accuracies_16_top5, label='VGG16')
plt.plot(train_accuracies_50_top5, label='Resnet50')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Resnet50 vs VGG16 Top 5 Training Accuracy')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(test_accuracies_16_top1, label='VGG16')
plt.plot(test_accuracies_50_top1, label='Resnet50')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Resnet50 vs VGG16 Top 1 Validation Accuracy')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(test_accuracies_16_top5, label='VGG16')
plt.plot(test_accuracies_50_top5, label='Resnet50')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Resnet50 vs VGG16 Top 5 Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.legend()
plt.show()

print(f"The best validation accuracy of VGG16 is {max(test_accuracies_16_top1): .2f} %")
print(f"The best validation accuracy of ResNet50 is {max(test_accuracies_50_top1): .2f} %")