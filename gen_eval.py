from unittest import result
import torch
import numpy as np
from Model.VideoSWIN import VideoSWIN3D
from configuration import build_config
#from my_dataloader import TinyVIRAT_dataset
from dataloader2 import TinyVirat
from torch.utils.data import  DataLoader
from tqdm import tqdm
import os

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
    print("Predictions: ",pred)
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
rmap= {}
with open('answer.txt', 'w') as wid:
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_generator)):
            
            inputs = inputs.cuda()

            ##### For dataloader 2 #######
            #squeeze clips dimension
            inputs = torch.squeeze(inputs,dim=1)
            video_id = targets[0]['path'][0]
            video_id = video_id.split('.')[0]


            predictions = model(inputs.float())
            
            #Get predicted labels for this video sample
            labels = compute_labels(predictions,inf_threshold)
            
            str_labels = str(labels)
            str_labels = str_labels.replace("[","")
            str_labels = str_labels.replace("]","")
            str_labels = str_labels.replace(",","")
            result_string = str(video_id) +' '+ str_labels

            #Add result to res dictionary
            rmap[video_id] = result_string
            count+=1
    
    #Add remaining video id labels in the answer.txt
    for id in range(6097):
        vid_id = str(id).zfill(5)
        if vid_id not in rmap:
           result_string = "{} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0".format(vid_id)
        else:
            result_string = rmap[vid_id]
        print("Result String: ",result_string)
        wid.write(result_string + '\n')
            
print(f"Total Samples: {count}")
