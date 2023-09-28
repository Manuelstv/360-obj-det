from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

def plot_bfov2(erp_image, bfov, class_color):
    center_x, center_y, phi, theta, h, w = bfov[:6]
    height, width, _ = erp_image.shape
    w_erp = int(w / 360 * width)
    h_erp = int(h / 180 * height)
    x1, y1 = center_x - w_erp // 2, center_y - h_erp // 2
    x2, y2 = center_x + w_erp // 2, center_y + h_erp // 2
    x1, x2 = np.clip([x1, x2], 0, width - 1)
    y1, y2 = np.clip([y1, y2], 0, height - 1)
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=class_color)

# Import the ERP image
erp_image_path = '/home/manuel/pesquisa-mestrado/images/7fzxZ.jpg'
erp_image = np.array(Image.open(erp_image_path))

# Read the JSON file
with open('/home/manuel/pesquisa-mestrado/annotations/7fzxZ.json', 'r') as f:
    data = json.load(f)

# Generate a color map for classes
unique_classes = set(data['class'])
num_classes = len(unique_classes)
colors = cm.rainbow(np.linspace(0, 1, num_classes))
class_to_color = {cls: colors[i] for i, cls in enumerate(unique_classes)}

# Plot the ERP image
plt.imshow(erp_image)
plt.axis('off')

# Plot each BFoV
for i, bfov in enumerate(data['boxes']):
    class_id = data['class'][i]
    class_color = class_to_color[class_id]
    plot_bfov(erp_image, bfov, class_color)

# Show the plot
plt.show()

