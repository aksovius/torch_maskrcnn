import torch
import torch.utils.data
from dataset import Dataset
from engine import train_one_epoch, evaluate
import utils
from model import *

DATASET = './dataset'

dataset = Dataset(DATASET, get_transform(train=True))
dataset_test = Dataset(DATASET, get_transform(train=False))

# split the dataset
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, 
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, 
    collate_fn=utils.collate_fn)

device = torch.device('cuda')

# background and rust
num_classes = 2

model = get_mobilnet_backbone(num_classes)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# decreases the learning rate by 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# train
from torch.optim.lr_scheduler import StepLR
num_epochs = 50

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    torch.save({
            'epoch': epoch, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            }, './model/custom_class_epoch_' + str(epoch) + '_.pth')
    lr_scheduler.step()
    evaluate(model, data_loader_test, device=device)