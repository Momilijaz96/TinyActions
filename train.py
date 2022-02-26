import torch
import numpy as np
from Model.ViViT_FE import ActRecogTransformer
from val_ncrc import validation
from utils.visualize import get_plot
import pickle
from asam import ASAM, SAM
from timm.loss import LabelSmoothingCrossEntropy


exp='Exp28n'

#CUDA for PyTorch
print("Using CUDA....")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
print("Creating params....")
params = {'batch_size':16,
          'shuffle': False,
          'num_workers': 4}
max_epochs = 250

# Generators
pose2id,labels,partition=PreProcessing_ncrc.preprocess()

print("Creating Data Generators...")

training_set = Poses3d_Dataset( data='ncrc',list_IDs=partition['train'], labels=labels, pose2id=pose2id,partition='train',modality='Mocap+Meditag')
training_generator = torch.utils.data.DataLoader(training_set, **params) #Each produced sample is  200 x 59 x 3

validation_set = Poses3d_Dataset(data='ncrc',list_IDs=partition['test'], labels=labels, pose2id=pose2id, partition='test',modality='Mocap+Meditag')
validation_generator = torch.utils.data.DataLoader(validation_set, **params) #Each produced sample is 6000 x 229 x 3

#Define model
print("Initiating Model...")
joints=29
stochastic_depth=0.2
drop_rate=0.
attn_drop_rate=0.
num_frames = 200
model=ActRecogTransformer(num_frames=num_frames,num_joints=joints,num_classes=6,drop_rate=drop_rate,attn_drop_rate=attn_drop_rate,drop_path_rate=stochastic_depth)
model=model.to(device)


#Define loss and optimizer
lr=0.01
wt_decay=5e-4
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=wt_decay)
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wt_decay)

#ASAM
rho=0.5
eta=0.01
minimizer = ASAM(optimizer, model, rho=rho, eta=eta)

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(minimizer.optimizer, max_epochs)

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
        #print("labels: ",targets.shape)
        predictions = model(inputs.float())
        #print("prediction size: ",predictions.shape)
        batch_loss = criterion(predictions, targets)
        batch_loss.mean().backward()
        minimizer.ascent_step()

        # Descent Step
        criterion(model(inputs.float()), targets).mean().backward()
        minimizer.descent_step()

        with torch.no_grad():
            loss += batch_loss.sum().item()
            accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
        cnt += len(targets)
    loss /= cnt
    accuracy *= 100. / cnt
    print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
    epoch_loss_train.append(loss)
    epoch_acc_train.append(accuracy)
    scheduler.step()

    #Test
    accuracy,loss,best_accuracy = validation(model,validation_generator)
    
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
