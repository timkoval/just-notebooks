import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import shutil
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchmetrics.classification import MulticlassF1Score


class GarbageDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, multiplier=1.0):
        self.image_paths = []
        self.classes = ("glass", "paper", "cardboard", "plastic", "metal", "trash")
        self.transform = transform
        self._unarchive()
        random.seed(42)
        random.shuffle(self.image_paths)
        self.image_paths = self.image_paths[: int(multiplier * len(self.image_paths))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        label = os.path.normpath(image_path).split(os.sep)[-2]
        num_label = self.classes.index(label)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, num_label

    def _unarchive(self):
        archive_path = os.path.join(os.getcwd(), "data", "garbage-classification.zip")
        extract_path = os.path.dirname(os.path.abspath(archive_path))
        self.images_dir = os.path.join(
            extract_path, "Garbage classification", "Garbage classification"
        )
        if len(os.listdir(extract_path)) <= 1:
            shutil.unpack_archive(extract_dir=extract_path, filename=archive_path)
        for class_dir in os.listdir(self.images_dir):
            for filename in os.listdir(os.path.join(self.images_dir, class_dir)):
                image_path = os.path.join(self.images_dir, class_dir, filename)
                self.image_paths.append(image_path)


def imshow(samples, predicted_labels=(), cols=5):
    images, labels = samples
    rows = len(samples[0]) // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i, image in enumerate(images):
        predicted_label = predicted_labels[i] if predicted_labels else labels[i]
        color = "green" if labels[i] == predicted_label else "red"
        ax.ravel()[i].imshow(image.T)
        ax.ravel()[i].set_title(
            train_dataset.classes[int(predicted_label)], color=color
        )
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def get_val_f1():
    """Func to get F1 score on a validation dataset."""
    model.eval()
    score = 0.0
    total = 0.0
    MetricClass = MulticlassF1Score(num_classes=len(val_dataset.classes)).to(device)

    with torch.no_grad():
        for images, labels in val_data:
            images = Variable(images.to(device, dtype=torch.float))
            labels = Variable(labels.to(device))

            outputs = model(images)
            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            score += MetricClass(predictions, labels)

    score = score / len(val_data)
    return score


def train(num_epochs):
    """Training function."""
    max_score = 0.0
    print("The model will be running on", device, "device")
    model.to(device)

    for epoch in range(num_epochs):
        current_loss = 0.0
        for index, (images, labels) in enumerate(tqdm(train_data), 0):
            images = Variable(images.to(device, dtype=torch.float))
            labels = Variable(labels.to(device))

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
            if index % 1000 == 999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {index + 1:5d}] loss: {current_loss / 1000:.3f}")
                current_loss = 0.0
        scheduler.step()
        f1_score = get_val_f1()
        print(f"Epoch: {epoch+1}    F1 Score: {f1_score}")

        if f1_score > max_score:
            path = "./model.pth"
            torch.save(model.state_dict(), path)
            max_score = f1_score


def test():
    """Testing function."""
    model.to(device)
    score = 0.0
    total = 0.0
    MetricClass = MulticlassF1Score(num_classes=len(val_dataset.classes)).to(device)

    with torch.no_grad():
        for images, labels in test_data:
            images = Variable(images.to(device, dtype=torch.float))
            labels = Variable(labels.to(device))

            outputs = model(images)
            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            score += MetricClass(predictions, labels)

    score = score / len(val_data)
    print(f"Testing F1 Score: {score}")
