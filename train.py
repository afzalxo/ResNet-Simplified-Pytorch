import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from resnet import resnet18

dev = torch.device('cuda:0')

def train_epoch(model, criterion, optimizer, trainloader):
    model.train()

    epoch_loss = .0
    matches = 0
    for samp, tru in trainloader:
        optimizer.zero_grad()
        res = model(samp.to(dev))
        loss = criterion(res, tru.to(dev))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        #Train Acc
        _, pred = torch.max(res, 1)
        matches += torch.sum(torch.squeeze(pred.eq(tru.to(dev).data.view_as(pred))))
    #print('Matches: {}, Len Trainloader: {}, Batchsize: {}'.format(matches, len(trainloader), trainloader.batch_size))
    epoch_acc = matches / float(len(trainloader)*trainloader.batch_size)
    avg_epoch_loss = epoch_loss/len(trainloader)
    return avg_epoch_loss, epoch_acc

def val_epoch(model, criterion, valloader):
    model.eval()

    epoch_loss = .0
    matches = 0
    for samp, tru in valloader:
        res = model(samp.to(dev))
        loss = criterion(res, tru.to(dev))
        epoch_loss += loss.item()
        #Acc
        _, pred = torch.max(res, 1)
        matches += torch.sum(torch.squeeze(pred.eq(tru.to(dev).data.view_as(pred))))
    epoch_acc = matches / float(len(valloader)*valloader.batch_size)
    avg_epoch_loss = epoch_loss/len(valloader)
    return avg_epoch_loss, epoch_acc

def main():
    num_epochs = 200
    lr = 0.1
    transforms_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((.4914,.4822,.4465), (.2023,.1994,.2010))]
            )
    transforms_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((.4914,.4822,.4465),(.2023, .1994, .2010))
        ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_train)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=4)
    classes=('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model = resnet18().to(dev)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark=True
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, weight_decay=5e-4, lr=lr)
    criterion = torch.nn.CrossEntropyLoss().to(dev)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(1, num_epochs+1):
        train_loss, train_acc = train_epoch(model, criterion, optimizer, trainloader)  
        print(f'Epoch: {epoch}, Train Loss: {train_loss}, Train Acc: {train_acc}')
        val_loss, val_acc = val_epoch(model, criterion, valloader)  
        print(f'Epoch: {epoch}, Val Loss: {val_loss}, Val Acc: {val_acc}')
        scheduler.step()

if __name__=='__main__':
    main()
