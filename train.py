import argparse
import json
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np

# Function to load and preprocess the data
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define data transforms for training, validation, and test sets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    validation_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using DataLoader for batching
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return trainloader, validationloader, testloader, train_data.class_to_idx

# Function to create a pretrained model and replace the classifier
def create_model(arch, hidden_units):
    if arch == 'vgg':
        model = models.vgg16(pretrained=True)
    elif arch == 'resnet':
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError("Architecture not supported")

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    return model

# Function to train the model
def train_model(model, trainloader, validationloader, epochs, learning_rate, device):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation loss and accuracy
        model.eval()
        validation_loss = 0
        accuracy = 0

        with torch.no_grad():
            for inputs, labels in validationloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                validation_loss += batch_loss.item()

                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation loss: {validation_loss/len(validationloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(validationloader):.3f}")

# Function to save the trained model as a checkpoint
def save_checkpoint(model, train_data, save_dir):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'arch': 'vgg16',  # Modify with the actual architecture used
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, save_dir)

def main():
    parser = argparse.ArgumentParser(description="Train a deep learning model to classify images")
    parser.add_argument("data_dir", help="Path to the data directory")
    parser.add_argument("--arch", choices=["vgg", "resnet"], default="vgg", help="Architecture (vgg or resnet)")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--save_dir", default="checkpoint.pth", help="Directory to save the model checkpoint")

    args = parser.parse_args()

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    
    trainloader, validationloader, _, train_data = load_data(args.data_dir)
    model = create_model(args.arch, args.hidden_units)
    train_model(model, trainloader, validationloader, args.epochs, args.learning_rate, device)
    save_checkpoint(model, train_data, args.save_dir)

if __name__ == "__main__":
    main()
