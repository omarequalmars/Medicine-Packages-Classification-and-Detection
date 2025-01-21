import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import h5py
import numpy as np
from PIL import Image

# Define data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class MedicineDataset(torch.utils.data.Dataset):
    """Custom Dataset for loading medicine images from H5 file"""
    def __init__(self, h5_file, transform=None):
        self.h5_file = h5_file
        self.transform = transform
        
        with h5py.File(self.h5_file, 'r') as f:
            self.images = f['images'][:]
            self.labels = f['labels'][:]
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert HWC to CHW format if needed
        if len(image.shape) == 3 and image.shape[-1] == 3:  # Check if image is in HWC format
            image = np.transpose(image, (2, 0, 1))  # Convert to CHW
            
        # Convert to PIL Image for transforms
        print(idx)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
            
        return image, label


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
    return model

def main():
    # Load data
    train_dataset = MedicineDataset(r'splits\train.h5', transform=data_transforms['train'])
    val_dataset = MedicineDataset(r'splits\val.h5', transform=data_transforms['val'])
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    }
    
    # Initialize model
    model = resnet18()
    num_classes = len(set(train_dataset.labels))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Train model
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=25)
    
    # Save model
    torch.save(model.state_dict(), 'medicine_classifier.pth')

if __name__ == '__main__':
    main()
