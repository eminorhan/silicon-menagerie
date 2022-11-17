import torch
from utils import load_model

# load model
model_name = 'dino_say_vitb14'
model = load_model(model_name)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()
print(model)

# visualize attention heads
from utils import preprocess_image, visualize_attentions
img_path = "imgs/img_0.jpg"
img_size = 1400  # nice and big
patch_size = 14
save_name = model_name + '_' + 'img_0'
img = preprocess_image(img_path, img_size)
with torch.no_grad():
    # this will visualize all attention heads for this particular image
    visualize_attentions(model, img, patch_size, save_name, device)