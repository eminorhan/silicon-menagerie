import torch
from utils import load_model

model_name = 'mae_s_vitb16'

model = load_model(model_name)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

print(model)