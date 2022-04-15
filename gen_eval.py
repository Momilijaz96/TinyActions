import torch
import numpy as np
from Model.VideoSWIN import VideoSWIN3D
from configuration import build_config
from dataloader import TinyVirat, VIDEO_LENGTH, TUBELET_TIME, NUM_CLIPS
from torch.utils.data import  DataLoader
from tqdm import tqdm

import os

exp='e19_val'

#Make exp dir
if not os.path.exists('evals/'+exp+'/'):
    os.makedirs('evals/'+exp+'/')
PATH='evals/'+exp+'/'


def compute_labels(pred,inf_th):
    
    pred = pred
    
    #Pass pred through sigmoid
    pred = torch.sigmoid(pred)
    pred = pred.cpu().data.numpy()

    #Use inference threshold to get one hot encoded labels
    res = pred > inf_th
    #print(res)
    pred = list(map(int, res[0])) 
    
    #Compute equal labels
    return pred

#CUDA for PyTorch
print("Using CUDA....")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


# Training Parameters
shuffle = True
print("Creating params....")
params = {'batch_size':1,
          'shuffle': shuffle,
          'num_workers': 2}

inf_threshold = 0.5

#Data Generators
dataset = 'TinyVirat'
cfg = build_config(dataset)
skip_frames = 2

test_dataset = TinyVirat(cfg, 'test', 1.0, num_frames = TUBELET_TIME,
                     skip_frames=2, input_size=120)
test_generator = DataLoader(test_dataset,1, shuffle=False, num_workers=0)

#Define model
print("Initiating Model...")

ckpt_path = '/home/mo926312/Documents/TinyActions/Slurm_Scripts/'+'exps/exp_19/19_best_ckpt.pt'
model = VideoSWIN3D()
model.load_state_dict(torch.load(ckpt_path))
model=model.to(device)

count = 0
print("Begin Evaluadtion....")
model.eval()

with open('answer.txt', 'w') as wid:
    vid_id = 0
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(tqdm(test_generator)):
            inputs = inputs.cuda()
            #print(inputs.shape)
            inputs  = torch.squeeze(inputs,dim=0) #To remove extra clips dimension
            predictions = model(inputs.float())
            
            #Get predicted labels for this video sample
            labels = compute_labels(predictions,inf_threshold)
            
            #Write video id and labels in file
            vid_id+=1

            str_labels = str(labels)
            str_labels.replace("[","")
            str_labels.replace("]","")
            str_labels.replace("[","")
            str_labels.replace("]","")

            result_string = str(vid_id) + str_labels
            print("Result String: ",result_string)
            wid.write(result_string + '\n')
            count+=1

print(f"Total Samples: {count}")
