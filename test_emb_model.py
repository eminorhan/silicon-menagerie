import os
import torch
from utils import load_model

# load model
model_name = 'dino_y_vitb14'
model = load_model(model_name)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

# visualize attention heads
from utils import preprocess_image, visualize_attentions

img_size = 1400  # nice and big
patch_size = 14
img_dir = "imgs_2"

for img_file in os.listdir(img_dir):
    if img_file.endswith(".jpg"):
        img_path = os.path.join(img_dir, img_file)
        save_name = model_name + '_' + img_path.split("/")[-1]
        img = preprocess_image(img_path, img_size)
        with torch.no_grad():
            # separate_heads=True to visualize all attention heads separately for this particular image
            visualize_attentions(model, img, patch_size, save_name, device, threshold=None, separate_heads=False)