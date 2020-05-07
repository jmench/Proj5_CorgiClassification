#Author: Jordan Menchen

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

def main():

  train_set = './data/training_data'
  test_set = './data/testing_data'
  classes = ['pembroke', 'cardigan']

  learn_rate = 0.001
  num_epochs = 30
  batch_size = 4

  TRANSFORM_IMG = transforms.Compose(
    [transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  train_data = torchvision.datasets.ImageFolder(root=train_set, transform=TRANSFORM_IMG)
  train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=2)
  test_data = torchvision.datasets.ImageFolder(root=test_set, transform=TRANSFORM_IMG)
  test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2) 

  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(3, 6, 5)
      self.pool = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(6, 16, 5)
      self.fc1 = nn.Linear(16 * 5 * 5, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = x.view(-1, 16 * 5 * 5)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x

  net = Net()

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=learn_rate)

  for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      if i % 30 == 29:    # print every 2000 mini-batches
        print('[epoch: %d, batch: %3d] loss: %.3f' %
          (epoch + 1, i + 1, running_loss / 30))
        running_loss = 0.0

  print('Finished Training')

  correct = 0
  total = 0
  with torch.no_grad():
    for data in test_data_loader:
      images, labels = data
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print('Accuracy of the network: %d %%' % (100 * correct / total))

  class_correct = list(0. for i in range(2))
  class_total = list(0. for i in range(2))
  with torch.no_grad():
    for data in test_data_loader:
      images, labels = data
      outputs = net(images)
      _, predicted = torch.max(outputs, 1)
      c = (predicted == labels).squeeze()
      for i in range(4):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1

  for i in range(2):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

if __name__ == '__main__':
    main()