import torch
import cv2
import numpy as np
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation

class MiDaSModel:
    def __init__(self):
        model_name = "Intel/dpt-beit-base-384"
        self.processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def infer_image(self, img_rgb):
        img_pil = Image.fromarray(img_rgb)
        inputs = self.processor(images=img_pil, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=img_pil.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        
        output = prediction.squeeze().cpu().numpy()
        return output

midas_model = MiDaSModel()
