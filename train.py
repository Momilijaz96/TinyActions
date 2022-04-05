import torch
import numpy as np
from Model.SpatialPerceiver_frame import Spatial_Perceiver
from configuration import build_config
from dataloader2 import TinyVirat, VIDEO_LENGTH, TUBELET_TIME, NUM_CLIPS
from asam import ASAM, SAM

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.visualize import get_plot
from sklearn.metrics import accuracy_score
import os

exp='12'

#Make exp dir
if not os.path.exists('exps/exp_'+exp+'/'):
    os.makedirs('exps/exp_'+exp+'/')
PATH='exps/exp_'+exp+'/'


def compute_accuracy(pred,target,inf_th):
    pred = pred
    target = target.cpu().data.numpy()
    #Pass pred through sigmoid
    pred = torch.sigmoid(pred)
    pred = pred.cpu().data.numpy()
    #Use inference throughold to get one hot encoded labels
    pred = pred > inf_th

    #Compute equal labels
    return accuracy_score(pred,target)

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
inf_threshold = 0.6
print(params)

#Data Generators
dataset = 'TinyVirat'
cfg = build_config(dataset)
skip_frames=2

train_dataset = TinyVirat(cfg, 'train', 1.0, num_frames = TUBELET_TIME, skip_frames=2, input_size=128)
training_generator = DataLoader(train_dataset,**params)

val_dataset = TinyVirat(cfg, 'val', 1.0, num_frames = TUBELET_TIME, skip_frames=2, input_size=128)
validation_generator = DataLoader(val_dataset, **params)

#Define model
print("Initiating Model...")


model=Spatial_Perceiver()
model=model.to(device)

#Define loss and optimizer
lr=0.01
wt_decay=5e-4
criterion=torch.nn.BCEWithLogitsLoss() #CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wt_decay)
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wt_decay)


#ASAM
rho=0.5
eta=0.01
minimizer = ASAM(optimizer, model, rho=rho, eta=eta)


# Learning Rate Scheduler
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)

#TRAINING AND VALIDATING
epoch_loss_train=[]
epoch_loss_val=[]
epoch_acc_train=[]
epoch_acc_val=[]

#Label smoothing
#smoothing=0.1
#criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)

best_accuracy = 0.
print("Begin Training....")
for epoch in range(max_epochs):
    
    #Train
    model.train()
    loss = 0.
    accuracy = 0.
    cnt = 0.

    for batch_idx, (inputs, targets) in enumerate(tqdm(training_generator)):
        inputs = inputs.to(device)
        #print("Inputs shape : ",inputs.shape)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Ascent Step
        predictions = model(inputs.float())
        batch_loss = criterion(predictions, targets)
        batch_loss.mean().backward()
        minimizer.ascent_step()

        # Descent Step
        criterion(model(inputs.float()), targets).mean().backward()
        minimizer.descent_step()

        with torch.no_grad():
            loss += batch_loss.sum().item()
            accuracy +=  compute_accuracy(predictions,targets,inf_threshold)
        cnt += len(targets) #number of samples

    loss /= cnt
    accuracy /= (batch_idx+1)
    print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
    epoch_loss_train.append(loss)
    epoch_acc_train.append(accuracy)
    #scheduler.step()

    #Test
    model.eval()
    loss = 0.
    accuracy = 0.
    cnt = 0.
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validation_generator):
            inputs = inputs.cuda(); #print("Validation sample: ",inputs,"target: ",targets)
            targets = targets.cuda()
            predictions = model(inputs.float())
            loss += criterion(predictions, targets).sum().item()
            accuracy += compute_accuracy(predictions,targets,inf_threshold)
            cnt += len(targets)
        loss /= cnt
        accuracy /= (batch_idx+1)

    if best_accuracy < accuracy:
       best_accuracy = accuracy; torch.save(model.state_dict(),PATH+exp+'_best_ckpt.pt');

    print(f"Epoch: {epoch}, Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}")
    epoch_loss_val.append(loss)
    epoch_acc_val.append(accuracy)
    #torch.save(model,exp+"_Last_epoch.pt")
    
    epoch_loss_val.append(loss)
    epoch_acc_val.append(accuracy)
    #torch.save(model,exp+"_Last_epoch.pt")


print(f"Best test accuracy: {best_accuracy}")
print("TRAINING COMPLETED :)")

#Save visualization
get_plot(PATH,epoch_acc_train,epoch_acc_val,'Accuracy-'+exp,'Train Accuracy','Val Accuracy','Epochs','Acc')
get_plot(PATH,epoch_loss_train,epoch_loss_val,'Loss-'+exp,'Train Loss','Val Loss','Epochs','Loss')

#Save trained model
torch.save(model,exp+"_ckpt.pt")
