import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.stats as st
from resnet import resnet18, resnet50

use_wandb = True
if use_wandb:
    import wandb
    wandb.init(project='resnet-test', name='full-200-resnet50')

dev = torch.device('cuda:0')

def reject_outliers(data, m=0.5):
    #data[np.abs(data)<0.001]=0.
    #data = data[abs(data) > 0.05]
    #return data
    return data[abs(data - np.mean(data)) > m * np.std(data)]

def lin_quant(data, m, n):
    data = np.clip(np.round(data*(2**n)), -2**(m+n-1), 2**(m+n-1))
    return data

def power2_quant(data, step, N):
    xq_hat = np.clip(np.round(np.log2(abs(data/step))), 0, 2**(N-1))-2
    xq = np.sign(data)*np.round(2**xq_hat)*step
    return xq

def twohot_quant(data, step, N, bla):
    xq_msb = power2_quant(data, step, N/2)/step
    xd = data/step - xq_msb
    xq_lsb = power2_quant(xd, step, N/2)/step
    xq = step * ((2**bla)* xq_msb + xq_lsb)
    return xq

def main():
    _train = False
    #Hyperparameters and augmentation adopted from https://github.com/kuangliu/pytorch-cifar
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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)
    model = resnet50().to(dev)
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, weight_decay=5e-4, lr=lr)
    criterion = torch.nn.CrossEntropyLoss().to(dev)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    if _train:
        best_loss = np.inf
        best_acc = .0
        for epoch in range(1, num_epochs+1):
            print(f'Training epoch {epoch}')
            train_loss, train_acc = train_epoch(model, criterion, optimizer, trainloader)  
            print(f'Epoch: {epoch}, Train Loss: {train_loss}, Train Acc: {train_acc}')
            val_loss, val_acc = val_epoch(model, criterion, valloader)  
            print(f'Epoch: {epoch}, Val Loss: {val_loss}, Val Acc: {val_acc}')
            if use_wandb:
                wandb.log({
                    'Train Loss':train_loss,
                    'Train Acc':train_acc,
                    'Val Loss':val_loss,
                    'Val Acc':val_acc, 
                    'Learning Rate':scheduler.get_lr()[0]
                    },step=epoch)
            if val_loss < best_loss:
                print(f'Val loss decreased from {best_loss} to {val_loss}. Saving Model...')
                torch.save(model.state_dict(), './models/resnet18.pth')
                best_loss = val_loss
                best_acc = val_acc
                print(f'Current best val acc: {best_acc}')
                if use_wandb:
                    wandb.log({'Best Acc': best_acc}, step = epoch)
            scheduler.step()
    else:
        model.load_state_dict(torch.load('./models/resnet18.pth')) 
        model.eval()
        val_loss, val_acc = val_epoch(model, criterion, valloader)
        print(f'Val Loss: {val_loss}, Val Acc: {val_acc}')
        for param_tensor in model.state_dict():
            if 'block1.2.conv2.weight' in param_tensor:
                print(param_tensor, "\t", model.state_dict()[param_tensor].size())
                #Calculate number of bins
                x = torch.flatten(model.state_dict()[param_tensor]).cpu().detach().numpy()
                #x = reject_outliers(x, m=1.5)
                plt.rcParams['font.size'] = '14'
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax2 = ax1.twiny()
                #qx = lin_quant(x, 1, 7)
                #qx = power2_quant(x, 0.025, 16)
                qx = twohot_quant(x, 0.015, 9,0.25)
                qx = list(set(qx))
                print(qx)
                yval = 1e2 * np.ones(len(qx))
                markerline,stemlines,_=ax2.stem(qx, yval, linefmt='k-.', markerfmt='kD')
                plt.setp(stemlines, 'linewidth', 0.5)
                plt.setp(markerline, markersize = 1.2)
                #plt.rc('font', size=18)
                #plt.rcParams.update({'font.size': 14})
                ax2.set_xlabel('Quantization Levels')
                #q25, q75 = np.percentile(x,[.25,.75])
                #bin_width = 2*(q75 - q25)*len(x)**(-1/3)
                #bins = round((x.max() - x.min())/bin_width)
                #print("Freedman–Diaconis number of bins:", bins)
                #plt.hist(qx, bins = 16, color='orange');
                #Fancier plt
                mn, mx = -0.01, 0.01#plt.xlim()
                #plt.xlim(mn, mx)
                kde_xs = np.linspace(mn, mx, 300)
                kde = st.gaussian_kde(x)
                ax1.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
                #plt.legend(loc="upper left")
                plt.ylabel('Probability')
                ax1.set_xlabel('Float Weight Values')
                #plt.xlabel('Value Bins')
                #plt.ylabel('# Occurances')
                plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
                fig.savefig('Figure.png')
                wandb.log({f'{param_tensor}': wandb.Image(Image.open('Figure.png'))})
                #wandb.log({f'{param_tensor}': plt.hist(torch.flatten(model.state_dict()[param_tensor]).cpu().detach().numpy(), bins=16)})
                fig.clf()
                plt.close()



def train_epoch(model, criterion, optimizer, trainloader):
    model.train()

    epoch_loss = .0
    matches = 0
    for samp, tru in trainloader:
        optimizer.zero_grad()
        res,_ = model(samp.to(dev))
        loss = criterion(res, tru.to(dev))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        #Train Acc
        _, pred = torch.max(res, 1)
        matches += torch.sum(torch.squeeze(pred.eq(tru.to(dev).data.view_as(pred))))
    epoch_acc = matches / float(len(trainloader)*trainloader.batch_size)
    avg_epoch_loss = epoch_loss/len(trainloader)
    return avg_epoch_loss, epoch_acc

def val_epoch(model, criterion, valloader):
    model.eval()

    epoch_loss = .0
    matches = 0
    for samp, tru in valloader:
        res,_ = model(samp.to(dev))
        loss = criterion(res, tru.to(dev))
        epoch_loss += loss.item()
        #Acc
        _, pred = torch.max(res, 1)
        matches += torch.sum(torch.squeeze(pred.eq(tru.to(dev).data.view_as(pred))))
    #if use_wandb:
    #    wandb.log({'b0':torch.flatten(activ[0]).cpu(), 'b1': torch.flatten(activ[1]).cpu(), 'b2': torch.flatten(activ[2]).cpu(), 'b3': torch.flatten(activ[3]).cpu(), 'b4': torch.flatten(activ[4]).cpu()})
    epoch_acc = matches / float(len(valloader)*valloader.batch_size)
    avg_epoch_loss = epoch_loss/len(valloader)
    return avg_epoch_loss, epoch_acc

if __name__=='__main__':
    main()
