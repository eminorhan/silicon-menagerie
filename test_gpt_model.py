import math
from gpt_utils import load_model, generate_images_freely, generate_images_from_half
from torchvision.utils import save_image

# load gpt & vq (encoder-decoder) models
model_name = 'say_gimel'
gpt_model, vq_model = load_model(model_name)


# ========== GENERATE UNCONDITIONAL SAMPLES ==========
n_samples = 36  # total number of samples to generate

x = generate_images_freely(gpt_model, vq_model, n_samples=n_samples)

# save generated images
save_image(x, "free_samples_from_{}.png".format(model_name), nrow=int(math.sqrt(n_samples)), padding=1, normalize=True)
# ============================================================


# ========== GENERATE CONDITIONAL SAMPLES ==========
n_imgs = 6  # number of images to condition on
n_samples_per_img = 6  # number of conditional samples per image
data_path = '/vast/eo41/SAY_1fps'  # replace this with desired data directory (we will use random images from this directory to condition on)

x = generate_images_from_half(gpt_model, vq_model, data_path, n_imgs=1, n_samples_per_img=2)

# save generated images
save_image(x, "conditional_samples_from_{}.png".format(model_name), nrow=n_imgs, padding=1, normalize=True)
# ============================================================