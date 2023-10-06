
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import json

# Define the CustomDataset class
class CustomDataset(Dataset):
    def __init__(self, annotations_dir, images_dir, transform=None):
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.annotations_files = sorted(os.listdir(self.annotations_dir))
        self.transform = transform
        self.label_map = {
            'airconditioner': 0,
            'backpack': 1,
            'bathtub': 2,
            'bed': 3,
            'board': 4,
            'book': 5,
            'bottle': 6,
            'bowl': 7,
            'cabinet': 8,
            'chair': 9,
            'clock': 10,
            'computer': 11,
            'cup': 12,
            'door': 13,
            'fan': 14,
            'fireplace': 15,
            'heater': 16,
            'keyboard': 17,
            'light': 18,
            'microwave': 19,
            'mirror': 20,
            'mouse': 21,
            'oven': 22,
            'person': 23,
            'phone': 24,
            'picture': 25,
            'potted plant': 26,
            'refrigerator': 27,
            'sink': 28,
            'sofa': 29,
            'table': 30,
            'toilet': 31,
            'tv': 32,
            'vase': 33,
            'washer': 34,
            'window': 35,
            'wine glass': 36,
        }

    def __len__(self):
        return len(self.annotations_files)

    def __getitem__(self, idx):
        annotation_file = self.annotations_files[idx]
        with open(os.path.join(self.annotations_dir, annotation_file)) as f:
            annotation_data = json.load(f)
        img_name = annotation_file.replace('.json', '.jpg')
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        boxes = [box[:4] for box in annotation_data['boxes']]
        labels = [self.label_map[box[6]] for box in annotation_data['boxes']]
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
        return image, target

# Define the YOLOLike model
class YOLOLike(nn.Module):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=37):
        super(YOLOLike, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Add more layers as needed
        )
        
        self.fc = nn.Sequential(
            nn.Linear(1, 1024),  # Incorrect dimensions, will fix in forward
            nn.ReLU(),
            nn.Linear(1024, grid_size * grid_size * (5 * num_boxes + num_classes))
        )
        
    def forward(self, x):
        x = self.conv(x)
        if self.fc[0].in_features == 1:  # Only do this once
            self.fc[0] = nn.Linear(x.view(x.size(0), -1).size(1), 1024).to(x.device)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Initialize the dataset, dataloader, model, and optimizer
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
dataset = CustomDataset('/home/mstveras/360-obj-det/annotations', '/home/mstveras/360-obj-det/images', transform=transform)


from torch.utils.data.dataloader import default_collate

def my_collate(batch):
    # Filter out the images that have no boxes (if any)
    batch = list(filter(lambda x: x is not None, batch))
    
    # Default collate the images
    images = default_collate([x[0] for x in batch])
    
    # Collect & collate the targets
    targets = [x[1] for x in batch]
    
    return images, targets

# DataLoader with custom collate function
data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=my_collate)

model = YOLOLike()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def yolo_loss(outputs, targets, S=7, B=2, C=37):
    lambda_coord = 5
    lambda_noobj = 0.5
    
    N = outputs.size(0)
    outputs = outputs.view(-1, S, S, B * 5 + C)
    
    localization_loss = 0
    confidence_loss = 0
    class_loss = 0
    
    for i in range(N):  # Loop over batches
        prediction = outputs[i]
        
        # Get the target for this batch element
        target = targets[i]
        
        # Bounding box targets
        box_target = target['boxes']
        
        # Class label targets
        class_target = target['labels']
        
        # Objectness prediction
        obj_prediction = torch.sigmoid(prediction[..., 4])
        
        # Bounding box predictions
        box_prediction = torch.sigmoid(prediction[..., :4])
        
        # Class label predictions
        class_prediction = torch.softmax(prediction[..., 5:], dim=-1)
        
        # Assuming each image could have multiple boxes
        for box, label in zip(box_target, class_target):
            # Compute the loss for each bounding box
            localization_loss += F.mse_loss(box_prediction, box.unsqueeze(0), reduction='sum')
            
            # Class loss (Cross-Entropy)
            class_loss += F.cross_entropy(class_prediction.view(-1, C), label.view(-1), reduction='sum')
        
        # Confidence loss (Binary Cross-Entropy)
        confidence_loss += F.binary_cross_entropy(obj_prediction, torch.ones_like(obj_prediction), reduction='sum')
    
    loss = lambda_coord * localization_loss + confidence_loss + lambda_noobj * (1 - torch.ones_like(obj_prediction)) * confidence_loss + class_loss
    return loss

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for images, targets in data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = yolo_loss(outputs, targets)  # Removed the wrapping in a list
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')