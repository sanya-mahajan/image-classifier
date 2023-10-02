import argparse
import torch
import json
import numpy as np
from PIL import Image
from torchvision import models, transforms

# Function to load the trained model from a checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = models.vgg16(pretrained=False)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

# Function to process an image for prediction
def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    return image

# Function to predict the class of an image
def predict(image_path, model, topk, category_names, device
