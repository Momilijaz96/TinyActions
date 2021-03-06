from matplotlib.pyplot import get
import torch
import numpy as np
#from Model.Sp_frame_shareweights import Spatial_Perceiver
#from Model.SpatialPerceiver_frame import Spatial_Perceiver
from Model.VideoSWIN import VideoSWIN3D
#from Model.ViViT_FE import ViViT_FE

from my_dataloader import TinyVIRAT_dataset
from Preprocessing import get_prtn
from asam import ASAM

from torch.utils.data import  DataLoader
from tqdm import tqdm
from utils.visualize import get_plot
from sklearn.metrics import accuracy_score
import os

exp='23'

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
params = {'batch_size':2,
          'shuffle': shuffle,
          'num_workers': 4}

max_epochs = 50
inf_threshold = 0.6
print(params)

######### Data Generators ########
train_list_IDs,train_labels,train_IDs_path = get_prtn('train')
train_dataset = TinyVIRAT_dataset(list_IDs=train_list_IDs,labels=train_labels,IDs_path=train_IDs_path)
training_generator = DataLoader(train_dataset,**params)

val_list_IDs,val_labels,val_IDs_path = get_prtn('val')
val_dataset = TinyVIRAT_dataset(list_IDs=val_list_IDs,labels=val_labels,IDs_path=val_IDs_path)
val_generator = DataLoader(val_dataset,**params)

#Define model
print("Initiating Model...")

#model=Spatial_Perceiver()
model = VideoSWIN3D()
model=model.to(device)


#Define loss and optimizer
lr=0.02
wt_decay=5e-4
criterion=torch.nn.BCEWithLogitsLoss() #CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wt_decay)
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wt_decay)


#ASAM
rho=0.55
eta=0.01
minimizer = ASAM(optimizer, model, rho=rho, eta=eta)

# Learning Rate Scheduler
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(minimizer.optimizer, max_epochs)

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
        targets = targets.to(device)
        #print("Input shape: ",inputs.shape)
        #print("Target shape:",targets.shape)

        optimizer.zero_grad()

        # Ascent Step
        predictions = (model(inputs.float()))
        #print("Predictions: ", predictions.shape)
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
        for batch_idx, (inputs, targets) in enumerate(val_generator):
            inputs = inputs.cuda(); #print("Val target: ",targets)
            targets = targets.cuda(); inputs  = torch.squeeze(inputs)
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
