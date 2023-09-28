import torch
import torch.nn as nn

class YOLOLike(nn.Module):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20):
        super(YOLOLike, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Add more layers as needed
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * (grid_size // 2) * (grid_size // 2), 1024),
            nn.ReLU(),
            nn.Linear(1024, grid_size * grid_size * (5 * num_boxes + num_classes))
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Initialize the model and optimizer
model = YOLOLike()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dummy input tensor (replace this with your actual data)
input_tensor = torch.randn(8, 3, 224, 224)  # Batch of 8 images, each of size 224x224 with 3 channels

# Forward pass
output_tensor = model(input_tensor)

# The output tensor shape should be [batch_size, 7 * 7 * (5 * num_boxes + num_classes)]
print(output_tensor.shape)

