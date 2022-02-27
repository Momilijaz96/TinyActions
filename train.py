import torch
import numpy as np
from Model.ViViT_FE import ViViT_FE
from configuration import build_config
from dataloader import TinyVirat
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.visualize import get_plot

exp='1'

#CUDA for PyTorch
print("Using CUDA....")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Training Parameters
shuffle = True
print("Creating params....")
params = {'batch_size':16,
          'shuffle': shuffle,
          'num_workers': 4}
max_epochs = 250

#Data Generators
dataset = 'TinyVirat'
cfg = build_config(dataset)
clip_length = 100
img_res = 128
skip_frames=2
train_dataset = TinyVirat(cfg, 'train', 1.0, num_frames=clip_length, skip_frames=2, input_size=img_res)
training_generator = DataLoader(train_dataset,**params)

val_dataset = TinyVirat(cfg, 'val', 1.0, num_frames=clip_length, skip_frames=2, input_size=img_res)
validation_generator = DataLoader(val_dataset, **params)

#Define model
print("Initiating Model...")
num_classes=26
vid_dim=(img_res,img_res,clip_length) #one sample dimension

tubelet_dim=(3,4,4,4)

spat_op='cls' #or GAP

model=ViViT_FE(vid_dim=vid_dim,num_classes=num_classes,tubelet_dim=tubelet_dim,spat_op=spat_op)
model=model.to(device)

#Define loss and optimizer
lr=0.01
wt_decay=5e-4
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=wt_decay)
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wt_decay)

'''
#ASAM
rho=0.5
eta=0.01
minimizer = ASAM(optimizer, model, rho=rho, eta=eta)
'''

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)

#TRAINING AND VALIDATING
epoch_loss_train=[]
epoch_loss_val=[]
epoch_acc_train=[]
epoch_acc_val=[]

#Label smoothing
#smoothing=0.1
#criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)

print("Begin Training....")
for epoch in range(max_epochs):
    # Train
    model.train()
    loss = 0.
    accuracy = 0.
    cnt = 0.
    for inputs, targets in training_generator:
        inputs = inputs.to(device); #print(inputs.shape)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Ascent Step
        predictions = model(inputs.float())
        batch_loss = criterion(predictions, targets)
        #batch_loss.mean().backward()
        #minimizer.ascent_step()
        batch_loss.backward()
        optimizer.step()

        # Descent Step
        #criterion(model(inputs.float()), targets).mean().backward()
        #minimizer.descent_step()

        with torch.no_grad():
            loss += batch_loss.sum().item()
            accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
        cnt += len(targets)
        scheduler.step()

    loss /= cnt
    accuracy *= 100. / cnt
    print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
    epoch_loss_train.append(loss)
    epoch_acc_train.append(accuracy)
    scheduler.step()

    #Test
    model.eval()
    loss = 0.
    accuracy = 0.
    cnt = 0.
    with torch.no_grad():
        for inputs, targets in validation_generator:
            inputs = inputs.cuda()
            targets = targets.cuda()
            predictions = model(inputs.float())
            loss += criterion(predictions, targets).sum().item()
            accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
            cnt += len(targets)
        loss /= cnt
        accuracy *= 100. / cnt

    if best_accuracy < accuracy:
       best_accuracy = accuracy

    print(f"Epoch: {epoch}, Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}")
    epoch_loss_val.append(loss)
    epoch_acc_val.append(accuracy)
    torch.save(model,exp+"_Last_epoch.pt")
    
    epoch_loss_val.append(loss)
    epoch_acc_val.append(accuracy)
    torch.save(model,exp+"_Last_epoch.pt")


print(f"Best test accuracy: {best_accuracy}")
print("TRAINING COMPLETED :)")

#Save visualization
get_plot(epoch_acc_train,epoch_acc_val,'Accuracy-'+exp,'Train Accuracy','Val Accuracy','Epochs','Acc')
get_plot(epoch_loss_train,epoch_loss_val,'Loss-'+exp,'Train Loss','Val Loss','Epochs','Loss')

#Save trained model
torch.save(model,exp+"_ckpt.pt")
