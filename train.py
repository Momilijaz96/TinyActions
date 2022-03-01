import torch
import numpy as np
from Model.ViViT_FE import ViViT_FE
from configuration import build_config
from dataloader import TinyVirat, VIDEO_LENGTH, TUBELET_TIME, NUM_CLIPS

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.visualize import get_plot
from sklearn.metrics import accuracy_score
exp='1'

def compute_accuracy(pred,target,inf_th):
    #Pass pred through sigmoid
    pred = torch.sigmoid(pred)

    #Use inference throughold to get one hot encoded labels
    pred = pred > inf_th

    #Compute equal labels
    return accuracy_score(pred,target)

#CUDA for PyTorch
print("Using CUDA....")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

#Data parameters
tubelet_dim=(3,TUBELET_TIME,4,4) #(ch,tt,th,tw)
num_classes=26  
img_res = 128
vid_dim=(img_res,img_res,VIDEO_LENGTH) #one sample dimension - (H,W,T)


# Training Parameters
shuffle = True
print("Creating params....")
params = {'batch_size':4,
          'shuffle': shuffle,
          'num_workers': 4}
max_epochs = 250
gradient_accumulations = 1
inf_threshold = 0.7

#Data Generators
dataset = 'TinyVirat'
cfg = build_config(dataset)
skip_frames=2

train_dataset = TinyVirat(cfg, 'train', 1.0, num_frames=tubelet_dim[1], skip_frames=2, input_size=img_res)
training_generator = DataLoader(train_dataset,**params)

val_dataset = TinyVirat(cfg, 'val', 1.0, num_frames=tubelet_dim[1], skip_frames=2, input_size=img_res)
validation_generator = DataLoader(val_dataset, **params)

#Define model
print("Initiating Model...")

spat_op='cls' #or GAP

model=ViViT_FE(vid_dim=vid_dim,num_classes=num_classes,tubelet_dim=tubelet_dim,spat_op=spat_op)
model=model.to(device)

#Define loss and optimizer
lr=0.01
wt_decay=5e-4
criterion=torch.nn.BCEWithLogitsLoss() #CrossEntropyLoss()
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
best_accuracy = 0.
print("Begin Training....")
for epoch in range(max_epochs):
    # Train
    model.train()
    loss = 0.
    accuracy = 0.
    cnt = 0.
    for batch_idx, (inputs, targets) in enumerate(tqdm(training_generator)):
        inputs = inputs.to(device)
        #print("Targets shape : ",targets.shape)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Ascent Step
        predictions = model(inputs.float()); #targets = torch.tensor(targets,dtype=torch.long); predictions = torch.tensor(predictions,dtype=torch.long)

        batch_loss = criterion(predictions, targets)


         # compute gradients of this batch.
        (batch_loss / gradient_accumulations).backward()
        # so each parameter holds its gradient value now,
        # and when we run `loss.backward()` again in next batch iteration,
        # then the previous gradient computed and the current one will be added.
        # this is the default behaviour of gradients in pytorch.

        if (batch_idx + 1) % gradient_accumulations == 0:
            optimizer.step()
            model.zero_grad()

        with torch.no_grad():
            loss += batch_loss.sum().item()
            accuracy +=  compute_accuracy(predictions,targets)
        cnt += len(targets) #number of samples
        scheduler.step()

    loss /= cnt; 
    accuracy /= batch_idx
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
            accuracy += ((predictions>inf_threshold)==targets).sum().item()  #(torch.argmax(predictions, 1) == targets).sum().item()
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
