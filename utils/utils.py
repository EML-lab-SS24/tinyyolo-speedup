from typing import List, Dict, Tuple

import torch
from torchinfo import summary
import time
import io
import gzip

import matplotlib.pyplot as plt
import numpy as np
import copy

from utils.loss import YoloLoss
from utils.ap import ap
from utils.train import validate
from utils.dataloader import VOCDataLoaderPerson

import seaborn as sns
import pandas as pd

def net_macs(model_class: torch.nn.Module, state_dict: Dict) -> int:
    net = model_class()
    net.load_state_dict(state_dict)
    res = summary(net, (1, 3, 32, 32), verbose=0)
    return res.total_mult_adds

def net_params(model_class: torch.nn.Module, state_dict: Dict) -> int:
    net = model_class()
    net.load_state_dict(state_dict)
    res = summary(net, (1, 3, 32, 32), verbose=0)
    return res.total_params

def net_time(model_class: torch.nn.Module, state_dict: Dict,
             iterations: int=5, device: str='cpu') -> float:

    torch_device = torch.device(device)
    net = model_class()
    net.load_state_dict(state_dict)
    net.eval()
    
    t = 0.0
    
    input = torch.rand(64, 3, 32, 32)
    input = input.to(torch_device)
    
    for _ in range(10):
        net.to(torch_device)
        torch.cuda.synchronize()
        out = net(input)
        torch.cuda.synchronize()
        torch.max(out)
    times = []
    for _ in range(iterations):
        t_start = time.time()
        torch.cuda.synchronize()
        out = net(input)
        torch.cuda.synchronize()
        t_end = time.time()
        torch.max(out)
        times.append(t_end - t_start)
        
    times = list(sorted(times, reverse=False))
        
    return np.mean(times[0:10])
    
def net_acc(model_class: torch.nn.Module,
            state_dict: Dict,
            testloader: torch.utils.data.DataLoader,
            device: str='cpu', batches: int=10) -> float:

    net = model_class()
    torch_device = torch.device(device)
    
    net.load_state_dict(state_dict)
    net.to(torch_device)
    correct_predictions = 0
    total_predictions = 0
    
    for idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(torch_device)
        outputs = net(inputs)
        
        correct_predictions +=(torch.argmax(outputs.cpu().detach(), axis=1) == 
                               targets.cpu().detach()).sum()
        
        total_predictions += int(targets.shape[0])
        
        if idx == (batches - 1):
            break
            
    accuracy = float(correct_predictions/total_predictions)
    return round(100*accuracy, 2)

def size_on_disk(state_dict: Dict) -> Tuple[int, int]:
    buff = io.BytesIO()
    torch.save(state_dict, buff)
    compressed_buff = gzip.compress(buff.getvalue(), compresslevel=9)
    return compressed_buff.__sizeof__(), buff.__sizeof__()


def plot(data, xlabel='Execution time', save_path='plot.png'):
   
    data = copy.deepcopy(data)
    for (x, y, label) in data:
        x = np.array(x)/max(x)
        plt.plot(x, y, label=label, alpha=0.5)
        plt.scatter(x, y, alpha=0.5)
        
    plt.ylabel('Accuracy')
    plt.xlabel(xlabel)
    plt.legend()
        
    plt.savefig(save_path)
    plt.show()


def identify_threshold(model_class, state_dict, num_classes=1, device=torch.device('cpu'), step_lengths=[0.5, 0.2, 0.1, 0.05, 0.025], anchors=[]):
    #state_dict = torch.load('models/configs/voc_pruned4.pt')
    #state_dict = torch.load('results/voc_finetuned_0_frozen_layers_67_epochs_3e-05_lr_0_decay.pt')
    
    net = model_class(num_classes=num_classes, anchors=anchors)
            
    net.load_state_dict(state_dict)
    
    times, losses = [], []
    
    criterion = YoloLoss(anchors=net.anchors)
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()))
    
    
    max_acc = -1
    max_filter = 0.5
    max_nms = 0.5
    
    for sl in range(1, len(step_lengths)):
        step_length = step_lengths[sl]
        prev_step_length = step_lengths[sl - 1]
        
        start_nms = max_nms - prev_step_length
        end_nms = max_nms + prev_step_length
        nms_threshold_steps = int((end_nms - start_nms) / step_length + 1)
    
        if (start_nms < 0):
            end_nms -= start_nms
            start_nms = 0
            
        if (end_nms > 1):
            start_nms -= (end_nms - 1)
            end_nms = 1
        
        start_filter = max_filter - prev_step_length
        end_filter = max_filter + prev_step_length
        filter_threshold_steps = int((end_filter - start_filter) / step_length + 1)
        
        if (start_filter < 0):
            end_filter -= start_filter
            start_filter = 0
            
        if (end_filter > 1):
            start_filter -= (end_filter + 1)
            end_filter = 1
        
        nms_thresholds = torch.linspace(start_nms, end_nms, steps=nms_threshold_steps)
        filter_thresholds = torch.linspace(start_filter, end_filter, steps=filter_threshold_steps)
        
        results = torch.zeros([filter_threshold_steps, nms_threshold_steps])
        
        max_acc = 0
        max_filter = 0
        max_nms = 0
        
        
        for i in range(len(filter_thresholds)):
            filter_threshold = filter_thresholds[i]
            #print(f"Filter Threshold: {filter_threshold}")
            for j in range(len(nms_thresholds)):
                nms_threshold = nms_thresholds[j]
                #print(f"NMS Threshold: {nms_threshold}")
        
                p, r, l, t = validate(net, VOCDataLoaderPerson(train=False, batch_size=1, shuffle=False), optimizer, criterion, device=device, filter_threshold=filter_threshold, nms_threshold=nms_threshold, batches=1000)
        
                acc = ap(p, r)
                
                
                #print('average time', t)
                #print('average test losses', l)
                #print('average precision', acc)
                losses.append(l)
                times.append(t)
        
                if (acc > max_acc):
                    max_acc = acc
                    max_filter = filter_threshold
                    max_nms = nms_threshold
        
                results[i][j] = acc
                #print("---------------------------------------------------------")
        
        print('Max accuracy:', max_acc)
        print('With filter threshold:', max_filter)
        print('With NMS threshold:', max_nms)
        cm = sns.light_palette("blue", as_cmap=True)
        x=pd.DataFrame(torch.round(100 * results, decimals=2))
        x=x.style.background_gradient(cmap=cm)
        display(x)
    return max_acc, max_filter, max_nms