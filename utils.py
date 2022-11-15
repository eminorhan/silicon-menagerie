"""
Util functions
"""
import os
import sys
import torch
from huggingface_hub import hf_hub_download

def load_model(model_name):

    alg, data, model_spec = model_name.split("_")

    # checks
    assert alg in ["dino", "mugs", "mae"], "Unrecognized algorithm!"
    assert data in ["say", "s", "a", "y", "imagenet_100", "imagenet_10", "imagenet_3", "imagenet_1"], "Unrecognized data!"
    assert model_spec in ["resnext50", "vitb14", "vitl16", "vitb16", "vits16"], "Unrecognized architecture!"

    if model_spec == "resnext50":
        arch, patch_size = "resnext50_32x4d", None
    elif model_spec == "vitb14":
        arch, patch_size = "vit_base", 14
    elif model_spec == "vitl16":
        arch, patch_size = "vit_large", 16
    elif model_spec == "vitb16":
        arch, patch_size = "vit_base", 16
    elif model_spec == "vits16":
        arch, patch_size = "vit_small", 16

    # download checkpoint from hf
    checkpoint = hf_hub_download(repo_id="eminorhan/"+model_name, filename=model_name+".pth")

    if alg == "dino":
        model = build_dino(arch, patch_size)
        load_dino(model, checkpoint, "teacher", arch, patch_size)

    print(model)

    return model

def load_dino(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

def build_dino(arch, patch_size):
    import vision_transformer_dino as vits
    from torchvision import models as torchvision_models

    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    if arch in vits.__dict__.keys():
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    # otherwise, we check if the architecture is in torchvision models
    elif arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[arch]()
        model.fc = torch.nn.Identity()
    else:
        print(f"Unknown architecture: {arch}")
        sys.exit(1)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    return model