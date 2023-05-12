import math
import argparse
from gpt_utils import load_model, generate_images_freely, generate_images_from_half
from torchvision.utils import save_image

parser = argparse.ArgumentParser('Generate conditional or unconditional samples from a GPT model', add_help=False)
parser.add_argument("--model_name", default="", type=str, help="GPT model name")
parser.add_argument("--mode", default="free", type=str, choices=['free', 'conditional'], help="Generation mode (free or conditional)")
parser.add_argument("--img_dir", default="", type=str, help="Image directory (needed only for conditional samples)")
parser.add_argument('--seed', default=1, type=int, help="Random seed")

args = parser.parse_args()
print(args)

# load gpt & vq (encoder-decoder) models
gpt_model, vq_model = load_model(args.model_name)

if args.mode == 'free':
    # generate unconditional samples
    n_samples = 25  # total number of samples to generate
    x = generate_images_freely(gpt_model, vq_model, n_samples=n_samples)
    # save generated images
    save_image(x, "free_samples_from_{}.png".format(args.model_name), nrow=int(math.sqrt(n_samples)), padding=1, normalize=True)
elif args.mode == 'conditional':
    # generate conditional samples
    n_imgs = 12            # number of images to condition on
    n_samples_per_img = 2  # number of conditional samples per image
    x = generate_images_from_half(gpt_model, vq_model, args.img_dir, n_imgs=n_imgs, n_samples_per_img=n_samples_per_img, seed=args.seed)
    # save original + generated images (separately, but feel free to change this)
    print(x.shape)
    save_image(x[:n_imgs,...], "original_imgs_from_{}_{}.png".format(args.model_name, args.seed), nrow=n_imgs, padding=1, normalize=True)
    save_image(x[n_imgs:,...], "conditional_samples_from_{}_{}.png".format(args.model_name, args.seed), nrow=n_imgs, padding=1, normalize=True)