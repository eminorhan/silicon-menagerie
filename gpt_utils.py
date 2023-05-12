import random
import numpy as np
import torch
import yaml
import gptmodel
from omegaconf import OmegaConf
from vqmodel import VQModel, GumbelVQ
from huggingface_hub import hf_hub_download


def get_available_models():
    available_models = [
        'say_gimel', 's_gimel', 'a_gimel', 'y_gimel',
        'imagenet100_gimel', 'imagenet10_gimel', 'imagenet1_gimel', 'imagenet100_alef', 'imagenet100_bet', 'imagenet100_dalet',
        'imagenet100+konkleiid_gimel', 'imagenet10+konkleiid_gimel', 'imagenet1+konkleiid_gimel',
        'say+konkleiid_gimel', 's+konkleiid_gimel', 'a+konkleiid_gimel', 'y+konkleiid_gimel',
        'say+konklenonvehicle_gimel', 's+konklenonvehicle_gimel', 'a+konklenonvehicle_gimel', 'y+konklenonvehicle_gimel',
        'imagenet100+konklenonvehicle_gimel', 'imagenet10+konklenonvehicle_gimel', 'imagenet1+konklenonvehicle_gimel',
        'konkleiid_gimel', 'konklenonvehicle_gimel'
        ]

    return available_models

def load_model(gpt_name):
    # check
    assert gpt_name in get_available_models(), "Unrecognized GPT model!"

    # parse identifier
    data, config = gpt_name.split("_")

    # assign corresponding vq model name
    data = data.split("+")[0]
    if data.startswith('konkle'):
        vq_name = "say_32x32_8192"
    elif data.startswith('imagenet'):
        vq_name = "imagenet_16x16_16384"
    else:
        vq_name = data + "_32x32_8192"        

    # download checkpoint from hf
    gpt_model_ckpt = hf_hub_download(repo_id="eminorhan/gpt_saycam", subfolder="gpt_pretrained_models", filename=gpt_name+".pt")
    vq_config_ckpt = hf_hub_download(repo_id="eminorhan/gpt_saycam", subfolder="vqgan_pretrained_models", filename=vq_name+".yaml")
    vq_model_ckpt = hf_hub_download(repo_id="eminorhan/gpt_saycam", subfolder="vqgan_pretrained_models", filename=vq_name+".ckpt")

    if data.startswith('imagenet'): 
        vocab_size, block_size = 16384, 255 
    else:
        vocab_size, block_size = 8192, 1023 

    # load gpt model
    gpt_config = gptmodel.__dict__['GPT_'+config](vocab_size, block_size)
    gpt_model = gptmodel.GPT(gpt_config)

    gpt_model_ckpt = torch.load(gpt_model_ckpt, map_location='cpu')
    gpt_model.load_state_dict(gpt_model_ckpt['model_state_dict'])

    # load vq model
    vq_config = load_config(vq_config_ckpt, display=True)
    vq_model = load_vqgan(vq_config, ckpt_path=vq_model_ckpt)

    return gpt_model, vq_model

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
        return config

def load_vqgan(config, ckpt_path=None, gumbel=False):
    if gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
        
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)

    return model.eval()

def generate_images_freely(gpt_model, vq_model, n_samples=1):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    gpt_model.to(device)
    gpt_model.eval()

    vq_model.to(device)
    vq_model.eval()

    # sample latents
    with torch.no_grad():
        s = gpt_model.sample_freely(n_samples=n_samples)

    # decode latents into images
    img_size, vq_dim = int(np.sqrt(gpt_model.model_config.block_size + 1)), 256
    z = vq_model.quantize.get_codebook_entry(s, (n_samples, img_size, img_size, vq_dim))
    x = vq_model.decode(z)

    return x

def generate_images_from_half(gpt_model, vq_model, img_dir, n_imgs=1, n_samples_per_img=2, seed=1):
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    set_seed(seed)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    gpt_model.to(device)
    gpt_model.eval()

    vq_model.to(device)
    vq_model.eval()

    # data preprocessing
    transform = Compose([Resize(256), CenterCrop(256), ToTensor()])
    dataset = ImageFolder(img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=n_imgs, shuffle=True, num_workers=4, pin_memory=True)

    for it, (imgs, _) in enumerate(loader):
        imgs = preprocess_vqgan(imgs)
        imgs = imgs.to(device)
        z, _, [_, _, indices] = vq_model.encode(imgs)
        indices = indices.reshape(n_imgs, -1)

        # only 1 iteration
        if it == 0:
            break

    n_samples = n_samples_per_img * n_imgs

    # sample latents
    with torch.no_grad():
        s = gpt_model.sample_from_half(indices, n_samples=n_samples_per_img)

    # decode latents into images
    img_size, vq_dim = int(np.sqrt(gpt_model.model_config.block_size + 1)), 256
    z = vq_model.quantize.get_codebook_entry(s, (n_samples, img_size, img_size, vq_dim))
    x = vq_model.decode(z)
    x[:, :, 126, :] = 0  # draw a line in the middle of image

    return x

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x