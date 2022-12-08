import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# import matplotlib.pyplot as plt


def load_data(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # If running on Windows and you get a BrokenPipeError, 
    # try setting the num_worker of torch.utils.data.DataLoader() to 0.
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


class BaselineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc = nn.Linear(16 * 5 * 5, 10)

    def forward(self, x):
        # input x: as CIFAR-10 data: torch.Size([batch_size, 3, 32, 32])
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)
        # output x: torch.Size([batch_size, classes_num])
        return x


# TODO define your net class here
class MyVGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, 3, padding=1)

        self.block2_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, 3, padding=1)

        self.block3_conv1 = nn.Conv2d(128, 256, 3, padding=1)
        self.block3_conv2 = nn.Conv2d(256, 256, 3, padding=1)

        self.block4_conv1 = nn.Conv2d(256, 512, 3, padding=1)
        self.block4_conv2 = nn.Conv2d(512, 512, 3, padding=1)

        self.block5_conv1 = nn.Conv2d(512, 512, 3, padding=1)
        self.block5_conv2 = nn.Conv2d(512, 512, 3, padding=1)

        self.fc1 = nn.Linear(512 * 1 * 1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.block1_conv1(x))
        x = self.relu(self.block1_conv2(x))
        x = self.pool(x)

        x = self.relu(self.block2_conv1(x))
        x = self.relu(self.block2_conv2(x))
        x = self.pool(x)

        x = self.relu(self.block3_conv1(x))
        x = self.relu(self.block3_conv2(x))
        x = self.pool(x)

        x = self.relu(self.block4_conv1(x))
        x = self.relu(self.block4_conv2(x))
        x = self.pool(x)

        x = self.relu(self.block5_conv1(x))
        x = self.relu(self.block5_conv2(x))
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class MyResNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def test(testloader, net, classes):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


if __name__ == "__main__":
    # load data
    trainloader, testloader, classes = load_data()
    # init model
    net = MyVGGNet()
    # use GPU acceleration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    # criterion and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    test(testloader, net, classes)

    ckpt_path = './cifar_net.pth'
    torch.save(net.state_dict(), ckpt_path)
