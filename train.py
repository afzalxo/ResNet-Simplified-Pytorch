import torch
import torchvision
import torchvision.transforms as transforms
from .resnet import resnet18

def train_epoch(model, criterion, optimizer, trainloader):
    model.train()

    epoch_loss = .0
    for samp, tru in trainloader:
        optimizer.zero_grad()
        res = model(samp.to(dev))
        loss = criterion(res, tru.to(dev))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss/len(trainloader)
    return avg_epoch_loss

def main():
    transform = transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((.5,.5,.5), (.5,.5,.5))]
            )
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torchvision.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    valloader = torchvision.utils.data.DataLoader(valset, batch_size=4, shuffle=False, num_workers=2)
    classes=('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model = resnet18()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 10):
        epoch_loss = train_epoch(model, criterion, optimizer, trainloader)  
        print('Epoch Loss: {epoch_loss}')

if __name__=='__main__':
    main()
