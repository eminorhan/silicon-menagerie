import torch
from utils import load_model

# load model
model_name = 'dino_ego4d-200h_vitb14'
model = load_model(model_name)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

# visualize attention heads
from utils import preprocess_image, visualize_attentions

img_size = 1400  # nice and big
patch_size = 14

for i in range(32):
    img_path = "imgs/img_{}.jpg".format(i)
    save_name = model_name + '_' + img_path.split("/")[-1]
    img = preprocess_image(img_path, img_size)
    with torch.no_grad():
        # separate_heads=True to visualize all attention heads separately for this particular image
        visualize_attentions(model, img, patch_size, save_name, device, threshold=None, separate_heads=False)