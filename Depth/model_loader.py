import torch
from depth_anything_v2.dpt import DepthAnythingV2

# Load the model
def load_model():
    model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))
    model.eval()

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)
    return model, device

# Load the model and device on import
model, device = load_model()
