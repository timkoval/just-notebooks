import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable


class Net(nn.Module):
    """A VGG-like net that we will use in this demo project."""

    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        #         self.conv8 = nn.Sequential(
        #             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        #             nn.BatchNorm2d(512),
        #             nn.ReLU())
        #         self.conv9 = nn.Sequential(
        #             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #             nn.BatchNorm2d(512),
        #             nn.ReLU())
        #         self.conv10 = nn.Sequential(
        #             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #             nn.BatchNorm2d(512),
        #             nn.ReLU(),
        #             nn.MaxPool2d(kernel_size = 2, stride = 2))
        #         self.conv11 = nn.Sequential(
        #             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #             nn.BatchNorm2d(512),
        #             nn.ReLU())
        #         self.conv12 = nn.Sequential(
        #             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #             nn.BatchNorm2d(512),
        #             nn.ReLU())
        #         self.conv13 = nn.Sequential(
        #             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #             nn.BatchNorm2d(512),
        #             nn.ReLU(),
        #             nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(7 * 7 * 256, 4096), nn.ReLU()
        )
        self.fc1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        #         out = self.conv8(out)
        #         out = self.conv9(out)
        #         out = self.conv10(out)
        #         out = self.conv11(out)
        #         out = self.conv12(out)
        #         out = self.conv13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def get_train_val_indices(validation_multiplier):
    """Train and validation splitting function."""
    indices = np.arange(len(train_set))
    np.random.shuffle(indices)
    splitting = int(np.floor(validation_multiplier * len(train_set)))
    return (indices[splitting:], indices[:splitting])


def imshow(img):
    """Image displaying function."""
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_accuracy():
    """Func to get accuracy on a validation dataset."""
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for images, labels in val_data:
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            outputs = model(images)
            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predictions == labels).sum().item()

    accuracy = 100 * accuracy / total
    return accuracy


def train(num_epochs):
    """Training function."""
    max_accuracy = 0.0
    print("The model will be running on", device, "device")
    model.to(device)

    for epoch in range(num_epochs):
        current_loss = 0.0
        current_accuracy = 0.0
        for index, (images, labels) in enumerate(tqdm(train_data), 0):
            images = Variable(images.to(device))
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
        accuracy = get_accuracy()
        print(f"Epoch: {epoch+1}    Accuracy: {accuracy}")

        if accuracy > max_accuracy:
            path = "./model.pth"
            torch.save(model.state_dict(), path)
            max_accuracy = accuracy


def test():
    """Testing function."""
    accuracy = 0.0
    total = 0.0
    with torch.no_grad():
        for images, labels in test_data:
            # images = Variable(images.to(device))
            # labels = Variable(labels.to(device))

            #             imshow(torchvision.utils.make_grid(images.cpu()))
            # print("Ground truth: ", ' '.join([classes[labels[idx]] for idx in range(config.data.batch_size)]))
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            accuracy += (predictions == labels).sum().item()
            # print("Prediction: ", ' '.join([classes[predictions[idx]] for idx in range(config.data.batch_size)]))

    accuracy = 100 * accuracy / total
    print(accuracy)
