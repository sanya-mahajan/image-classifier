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

def predict(image_path, model, topk, category_names, device):
    # Process the image
    image = process_image(image_path)
    image = image.unsqueeze(0).to(device)

    # Set the model to evaluation mode
    model.eval()
    
    # Make the prediction
    with torch.no_grad():
        output = model(image)

    # Calculate probabilities and top K classes
    probabilities = torch.exp(output)
    top_p, top_class = probabilities.topk(topk, dim=1)

    # Convert indices to classes
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_class[0].tolist()]

    # If category_names are provided, convert class indices to actual names
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_class_names = [cat_to_name[c] for c in top_classes]
    else:
        top_class_names = top_classes

    return top_p[0].tolist(), top_class_names

def main():
    parser = argparse.ArgumentParser(description="Image Classifier Prediction")
    parser.add_argument("image_path", help="Path to the image for prediction")
    parser.add_argument("checkpoint", help="Path to the trained model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", help="Path to a JSON file that maps category names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    # Determine the device (GPU or CPU)
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    # Load the model from the checkpoint
    model = load_checkpoint(args.checkpoint)
    model.to(device)

    # Perform prediction
    top_p, top_class_names = predict(args.image_path, model, args.top_k, args.category_names, device)

    # Print the results
    for i in range(args.top_k):
        print(f"Prediction {i+1}: Class='{top_class_names[i]}' | Probability={top_p[i]*100:.2f}%")

if __name__ == "__main__":
    main()