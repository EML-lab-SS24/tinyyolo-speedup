import torch
from typing import Dict
from utils.loss import YoloLoss
import numpy as np
import tqdm
from utils.ap import preprocess_for_ap, precision_recall_levels, ap, display_roc
from utils.yolo import nms, filter_boxes
import time
from utils.dataloader import VOCDataLoaderPerson

def train(net, trainloader: torch.utils.data.DataLoader, optimizer, criterion, device=torch.device('cpu')):
    
    net.to(device)

    net.train()
    
    train_running_loss = []
    print("------------------------Training-----------------------")
    for idx, (input, target) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        input = input.to(device)
        optimizer.zero_grad()
    
        #Yolo head is implemented in the loss for training, therefore yolo=False
        output = net(input, yolo=False).cpu()
        train_loss, _ = criterion(output, target)
        train_running_loss.append(train_loss.item())
    
        # writer.add_scalar("Loss/train", loss)
        train_loss.backward()
        optimizer.step()
        
    return net, np.mean(train_running_loss)
    
def validate(net, loader_test, optimizer, criterion, batches: int=350, device=torch.device('cpu'), filter_threshold=0.0375, nms_threshold=0.6):
    test_precision = []
    test_recall = []
    test_running_loss = []
    times = []
    
    net.to(device)
    net.eval()
    
    print("------------------------Validation-----------------------")
    with torch.no_grad():
        for idx, (input, target) in tqdm.tqdm(enumerate(loader_test), total=batches):
            input = input.to(device)
            t_start = time.time()
            net_output = net(input, yolo=False).cpu()

    
            output = preprocess_for_ap(net_output, net.anchors)
            #The right threshold values can be adjusted for the target application
            output = filter_boxes(output, filter_threshold)
            output = nms(output, nms_threshold)
            t_end = time.time()
            times.append(t_end - t_start)
            test_loss, _ = criterion(net_output, target)
            test_running_loss.append(test_loss.item())
            
            for i in range(len(output)):
                p, r = precision_recall_levels(target[i], output[i])
                test_precision.append(p)
                test_recall.append(r)
    
            if idx == batches:
                break
                
    return test_precision, test_recall, np.mean(test_running_loss), np.mean(times)

def longtrain(model_class: torch.nn.Module, state_dict: Dict={}, train_batch_size=64, test_batch_size=1, batches: int=350,
              device=torch.device('cpu'), num_classes=1, epochs=100, frozen_layers=[], lr=0.00005, weight_decay=0.0005,
              filter_threshold=0.1, nms_threshold=0.5, anchors=[]):
    # We define a tinyyolo network with only two possible classes
    net = model_class(num_classes=num_classes, anchors=anchors)

    if not state_dict == {}:
        #We load all parameters from the pretrained dict except for the last layer
        try:
            net.load_state_dict(state_dict)
        except:
            net.load_state_dict({k: v for k, v in state_dict.items() if not '9' in k}, strict=False)
    net.to(device)
    net.eval()
    
    # Definition of the loss
    criterion = YoloLoss(anchors=net.anchors)
    
    nr_frozen = len(frozen_layers)
    
    #We only train the last layer (conv9)
    for key, param in net.named_parameters():
        if any(x in key for x in frozen_layers):
            param.requires_grad = False
    
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)
    
    NUM_TEST_SAMPLES = batches
    NUM_EPOCHS = epochs
    test_AP = []
    last_sds = []
    avg_train_losses = []
    avg_test_losses = []
    early_stopping_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        
        trainloader = VOCDataLoaderPerson(train=True, batch_size=train_batch_size, shuffle=True)
        loader_test = VOCDataLoaderPerson(train=False, batch_size=test_batch_size, shuffle=True)
        
        net, train_running_loss = train(net, trainloader, optimizer, criterion, device)
    
        avg_train_losses.append(train_running_loss)

        test_precision, test_recall, test_running_loss, _ = validate(net, loader_test, optimizer, criterion, batches, device, filter_threshold, nms_threshold)
    
        #Calculation of average precision with collected samples
        avg_test_losses.append(test_running_loss)
        
        test_AP.append(ap(test_precision, test_recall))

        print_stats(train_running_loss, test_running_loss, test_AP[-1])
        print('test_precision', np.mean(test_precision))
        print('test_recall', np.mean(test_recall))
    
        last_sds.append(net.cpu().state_dict())
        if (len(last_sds) > 20):
            last_sds.pop(0)

    
        # Early Stopping
        #if epoch > 0 and abs(avg_test_losses[epoch] - avg_test_losses[epoch-1]) < 0.001:
        if (len(avg_test_losses) >= 20):
            if (np.mean(avg_test_losses[-20:-10]) < np.mean(avg_test_losses[-10:])):
            #early_stopping_counter += 1
            #if early_stopping_counter > 5:
                # stop training
                print("------------------------EARLY STOPPING------------------------")
                torch.save(net.cpu().state_dict(), f"results/voc_finetuned_{nr_frozen}_frozen_layers_{epoch+1}_epochs_{lr}_lr_{weight_decay}_decay.pt")
                with open(f'results/train_losses_finetuned_{nr_frozen}_frozen_layers_{epoch+1}_epochs_{lr}_lr_{weight_decay}_decay.txt', 'w') as f:
                    print(avg_train_losses, file=f)
                with open(f'results/test_losses_finetuned_{nr_frozen}_frozen_layers_{epoch+1}_epochs_{lr}_lr_{weight_decay}_decay.txt', 'w') as f:
                    print(avg_test_losses, file=f)
                with open(f'results/test_losses_finetuned_{nr_frozen}_frozen_layers_{epoch+1}_epochs_{lr}_lr_{weight_decay}_decay.txt', 'w') as f:
                    print(avg_test_losses, file=f)
                break
            #else:
                #early_stopping_counter = 0
    
        if (epoch + 1) % 25 == 0:
            display_roc(test_precision, test_recall)
    
        if (epoch + 1) % 15 == 0:
            # save checkpoint
            torch.save(net.state_dict(), f"results/voc_finetuned_{nr_frozen}_frozen_layers_{epoch+1}_epochs_{lr}_lr_{weight_decay}_decay.pt")
            # Save new losses and aps in files    
            with open(f'results/train_losses_finetuned_{nr_frozen}_frozen_layers_{epoch+1}_epochs_{lr}_lr_{weight_decay}_decay.txt', 'w') as f:
                print(avg_train_losses, file=f)
            with open(f'results/test_losses_finetuned_{nr_frozen}_frozen_layers_{epoch+1}_epochs_{lr}_lr_{weight_decay}_decay.txt', 'w') as f:
                print(avg_test_losses, file=f)
            with open(f'results/test_aps_finetuned_{nr_frozen}_frozen_layers_{epoch+1}_epochs_{lr}_lr_{weight_decay}_decay.txt', 'w') as f:
                print(test_AP, file=f)

    index = torch.argmax(torch.FloatTensor(avg_test_losses[-20:]))
    return last_sds[index]

def print_stats(train_loss, test_loss, test_ap):
    print('average train losses', train_loss)
    print('average test losses', test_loss)
    print('average precision', test_ap)
    