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
from torchmetrics.classification import BinaryF1Score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BlurDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename, blur, image_path = self.df.loc[idx].get(
            ["filename", "blur", "path_file"]
        )
        image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(image=image)["image"]
        return filename, image, blur


def unpack():
    archive_path = os.path.join(os.getcwd(), "data", "shift-cv-winter-2023.zip")
    extract_path = os.path.dirname(os.path.abspath(archive_path))
    shutil.unpack_archive(extract_dir=extract_path, filename=archive_path)


def imshow(samples, predicted_labels=(), cols=5):
    _, images, labels = samples
    rows = len(samples[0]) // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i, image in enumerate(images):
        predicted_label = predicted_labels[i] if predicted_labels else labels[i]
        color = "green" if labels[i] == predicted_label else "red"
        ax.ravel()[i].imshow(image.T)
        ax.ravel()[i].set_title(float(predicted_label), color=color)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def get_val_f1(model, val_data):
    """Func to get F1 score on a validation dataset."""
    model.eval()
    score = 0.0
    total = 0.0
    MetricClass = BinaryF1Score().to(device)

    with torch.no_grad():
        for _, images, labels in val_data:
            images = Variable(images.to(device, dtype=torch.float))
            labels = Variable(labels.to(device, dtype=torch.float))

            outputs = model(images).reshape(-1)
            predictions = outputs > 0.5
            total += labels.size(0)
            score += MetricClass(predictions, labels)

    score = score / len(val_data)
    return score


def train(model, train_data, val_data, num_epochs, loss_func, optimizer, scheduler):
    """Training function."""
    max_score = 0.0
    print("The model will be running on", device, "device")
    model.to(device)

    for epoch in range(num_epochs):
        current_loss = 0.0
        for index, (_, images, labels) in enumerate(tqdm(train_data), 0):
            images = Variable(images.to(device, dtype=torch.float))
            labels = Variable(labels.to(device, dtype=torch.float))

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_func(outputs.reshape(-1), labels)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
            if index % 1000 == 999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {index + 1:5d}] loss: {current_loss / 1000:.3f}")
                current_loss = 0.0
        scheduler.step()
        f1_score = get_val_f1(model, val_data)
        print(f"Epoch: {epoch+1}    F1 Score: {f1_score}")

        if f1_score > max_score:
            path = "./model.pth"
            torch.save(model.state_dict(), path)
            max_score = f1_score


def test(model, test_data):
    """Testing function."""
    model.to(device)
    filenames = []
    preds = []
    score = 0.0
    total = 0.0
    MetricClass = BinaryF1Score().to(device)

    with torch.no_grad():
        for names, images, labels in test_data:
            images = Variable(images.to(device, dtype=torch.float))
            labels = Variable(labels.to(device, dtype=torch.float))

            outputs = model(images).reshape(-1)
            predictions = outputs > 0.5
            total += labels.size(0)
            # score += MetricClass(predictions, labels)
            preds.extend(predictions.tolist())
            filenames.extend(names)

    # score = score / len(test_data)
    # print(f"Testing F1 Score: {score}")
    return filenames, preds
