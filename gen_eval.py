import torch
import numpy as np
from Model.VideoSWIN import VideoSWIN3D
from configuration import build_config
#from my_dataloader import TinyVIRAT_dataset
from dataloader2 import TinyVirat
from torch.utils.data import  DataLoader
from tqdm import tqdm
import os
from Preprocessing import get_prtn

exp='e22_val'

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

#Data Generators - mydataloader
'''
test_list_IDs,test_labels,test_IDs_path = get_prtn('test')
test_dataset = TinyVIRAT_dataset(list_IDs=test_list_IDs,labels=test_labels,IDs_path=test_IDs_path)
test_generator = DataLoader(test_dataset,**params)
'''
cfg = build_config('TinyVirat')
dataset = TinyVirat(cfg=cfg,data_split='test')
test_generator = DataLoader(dataset,**params)

#Define model
print("Initiating Model...")

ckpt_path = '/home/mo926312/Documents/TinyActions/Slurm_Scripts/'+'exps/exp_22/22_best_ckpt.pt'
model = VideoSWIN3D()
model.load_state_dict(torch.load(ckpt_path))
model=model.to(device)

count = 0
print("Begin Evaluadtion....")
model.eval()
with open('answer.txt', 'w') as wid:
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_generator)):
            
            inputs = inputs.cuda()

            ##### For dataloader 2 #######
            #squeeze clips dimension
            inputs = torch.squeeze(inputs,dim=1)
            video_id = targets[0]['path'][0]
            video_id = video_id.split('.')[0]

            print("video id: ",video_id)
            print("Inputs dim: ",inputs.shape)

            predictions = model(inputs.float())
            
            #Get predicted labels for this video sample
            labels = compute_labels(predictions,inf_threshold)
            
            str_labels = str(labels)
            str_labels = str_labels.replace("[","")
            str_labels = str_labels.replace("]","")
            str_labels = str_labels.replace(",","")

            result_string = str(video_id) +' '+ str_labels
            print("Result String: ",result_string)
            wid.write(result_string + '\n')
            count+=1

print(f"Total Samples: {count}")
