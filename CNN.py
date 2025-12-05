import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
This module demonstrates a complete workflow for training and evaluating a Convolutional Neural Network (CNN) 
using PyTorch on the CIFAR-10 dataset. It includes definition, training, evaluation and visualization.
"""

class Net(nn.Module):
    """
    Creates a Convolutional Neural Network (CNN) with two convolutional layers,
    followed by three fully connected layers. The network uses ReLU activation functions

    Architecture:
    - Convolutional Layer 1: Input channels = 3, Output channels = 6, Kernel size = 5
    - Max Pooling Layer: 2x2
    - Convolutional Layer 2: Input channels = 6, Output channels = 16, Kernel size = 5
    - Max Pooling Layer: 2x2
    - Fully Connected Layer 1: Input features = 400, Output features = 120
    - Fully Connected Layer 2: Input features = 120, Output features = 84
    - Fully Connected Layer 3: Input features = 84, Output features = 10

    Attributes:
    conv1,conv2 = Convloutional layers
    pool = Max pooling layer
    fc1,fc2,fc3 = Fully connected layers
    """
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    """
    Defines the forward propogation of the network. 
    Applies ReLU activation after each convolutional and fully connected layer,
    and max pooling after each convolutional layer.

    Args:
    x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32)
        
    Returns:
     torch.Tensor: Output logits of shape (batch_size, 10)

    """
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    """
    Displays an image tensor
    Unnormalizes the image and converts it to a numpy array for visualization.

    Returns:
    None
    """
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    """
    Training and Evaluation pipefline for CIFAR-10 classification

    1. Loads and normalizes CIFAR-10 Dataset
    2. Initalizes CNN Model
    3. Defines Loss function and Optimizer
    4. Trains the model for 2 epochs
    5. Evaluates the model on test data
    6. Visualizes some predictions

    Returns:
    None
    """


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))

    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    path = './cifar_net.pth'
    torch.save(net.state_dict(), path) 

    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    net = Net()
    net.load_state_dict(torch.load(path, weights_only=True))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                 for j in range(4)))
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

if __name__ == '__main__':
    main() 
