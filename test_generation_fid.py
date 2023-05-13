import os
import numpy as np
import argparse
import torch
from gpt_utils import load_model, preprocess_vqgan
from torchvision.utils import save_image
from torch_fidelity import calculate_metrics
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser('Generate conditional or unconditional samples from a GPT model', add_help=False)
parser.add_argument("--model_name", default="", type=str, help="GPT model name")
parser.add_argument("--task", default="labeled_s", type=str, choices=['labeled_s', 'konkle_iid', 'konkle_ood'], help="Task")
parser.add_argument("--img_dir", default="", type=str, help="Image directory (needed only for conditional samples)")
parser.add_argument('--subsample', default=False, action='store_true', help='whether to subsample the data')
parser.add_argument('--batch_size', default=12, type=int, help="Batch size")

args = parser.parse_args()
print(args)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# create image directories
org_dir = os.path.join('fids', args.task, args.model_name, 'imgs')
gen_dir = os.path.join('fids', args.task, args.model_name, 'gens')
os.makedirs(org_dir)
os.makedirs(gen_dir)

# load gpt & vq (encoder-decoder) models
gpt_model, vq_model = load_model(args.model_name)

gpt_model.to(device)
gpt_model.eval()

vq_model.to(device)
vq_model.eval()

# data preprocessing
transform = Compose([Resize(256), CenterCrop(256), ToTensor()])
dataset = ImageFolder(args.img_dir, transform=transform)

if args.subsample:
    from torch.utils.data.sampler import SubsetRandomSampler
    num_data = len(dataset)
    idxs = list(range(num_data))
    np.random.shuffle(idxs)
    idxs = idxs[:1024]
    sampler = SubsetRandomSampler(idxs)
    print('Num data:', len(idxs))
else:
    sampler = None
    print('Num data:', len(dataset))

loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, sampler=sampler)

# completions
for it, (imgs, _) in enumerate(loader):
    n_imgs = len(imgs)
    print('Num imgs:', n_imgs)
    imgs = preprocess_vqgan(imgs)
    imgs = imgs.to(device)
    z, _, [_, _, indices] = vq_model.encode(imgs)
    indices = indices.reshape(n_imgs, -1)

    # sample latents
    with torch.no_grad():
        s = gpt_model.sample_from_half(indices)

    # decode latents into images
    img_size, vq_dim = int(np.sqrt(gpt_model.model_config.block_size + 1)), 256
    z = vq_model.quantize.get_codebook_entry(s, (2 * n_imgs, img_size, img_size, vq_dim))
    x = vq_model.decode(z)
    print('Generation shape:', x.shape)

    # save images 
    for i in range(n_imgs):
        save_image(x[i, ...], os.path.join(org_dir, "image_{:04d}.jpeg".format(it * args.batch_size + i)), normalize=True)
        save_image(x[i+n_imgs, ...], os.path.join(gen_dir, "gener_{:04d}.jpeg".format(it * args.batch_size + i)), normalize=True)

metrics_dict = calculate_metrics(input1=org_dir, input2=gen_dir, isc=True, fid=True)
print(metrics_dict)

# save to file
f = open("{}_{}.txt".format(args.task, args.model_name), "w")
f.write(str(metrics_dict))
f.close()