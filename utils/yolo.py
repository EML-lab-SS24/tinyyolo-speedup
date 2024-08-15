import os
import os
import torch
from typing import List, Dict
from tqdm import tqdm
import time

from utils.loss import YoloLoss
from utils.ap import ap, precision_recall_levels, display_roc


def net_time(model_class, testloader):
    
    #----to-be-done-by-student-------------------
    t_start = time.time()
    x, y = next(iter(testloader))
    model_class(x)
    t_end = time.time()
    #----to-be-done-by-student-------------------
    t = t_end - t_start
    return t


def print_size_of_model(model):
    """ Prints the real size of the model """
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def iou(bboxes1, bboxes2):
    """ calculate iou between each bbox in `bboxes1` with each bbox in `bboxes2`"""
    px, py, pw, ph = bboxes1[...,:4].reshape(-1, 4).split(1, -1)
    lx, ly, lw, lh = bboxes2[...,:4].reshape(-1, 4).split(1, -1)
    px1, py1, px2, py2 = px - 0.5 * pw, py - 0.5 * ph, px + 0.5 * pw, py + 0.5 * ph
    lx1, ly1, lx2, ly2 = lx - 0.5 * lw, ly - 0.5 * lh, lx + 0.5 * lw, ly + 0.5 * lh
    zero = torch.tensor(0.0, dtype=px1.dtype, device=px1.device)
    dx = torch.max(torch.min(px2, lx2.T) - torch.max(px1, lx1.T), zero)
    dy = torch.max(torch.min(py2, ly2.T) - torch.max(py1, ly1.T), zero)
    intersections = dx * dy
    pa = (px2 - px1) * (py2 - py1) # area
    la = (lx2 - lx1) * (ly2 - ly1) # area
    unions = (pa + la.T) - intersections
    ious = (intersections/unions).reshape(*bboxes1.shape[:-1], *bboxes2.shape[:-1])
    
    return ious


def nms(filtered_tensor: List[torch.Tensor], threshold: float) -> List[torch.Tensor]:
    result = []
    for x in filtered_tensor:
        # Sort coordinates by descending confidence
        scores, order = x[:, 4].sort(0, descending=True)
        x = x[order]
        ious = iou(x,x) # get ious between each bbox in x

        # Filter based on iou
        keep = (ious > threshold).long().triu(1).sum(0, keepdim=True).t().expand_as(x) == 0

        result.append(x[keep].view(-1, 6).contiguous())
    return result


def filter_boxes(output_tensor: torch.Tensor, threshold) -> List[torch.Tensor]:
    b, a, h, w, c = output_tensor.shape
    x = output_tensor.contiguous().view(b, a * h * w, c)

    boxes = x[:, :, 0:4]
    confidence = x[:, :, 4]
    scores, idx = torch.max(x[:, :, 5:], -1)
    idx = idx.float()
    scores = scores * confidence
    mask = scores > threshold

    filtered = []
    for c, s, i, m in zip(boxes, scores, idx, mask):
        if m.any():
            detected = torch.cat([c[m, :], s[m, None], i[m, None]], -1)
        else:
            detected = torch.zeros((0, 6), dtype=x.dtype, device=x.device)
        filtered.append(detected)
    return filtered


def train(net_class: torch.nn.Module, num_classes: int, state_dict: Dict, loader: torch.utils.data.DataLoader, loader_test: torch.utils.data.DataLoader, writer=None, num_epochs: int=15, lr: float=0.001, conf_threshold: float=0.1, iou_threshold: float=0.5, device: str="cpu", freeze=False):
    if freeze:
        for key, param in net.named_parameters():
            if any(x in key for x in ['1', '2', '3', '4', '5', '6', '7']): # TODO: this does not contain 8, why not?
                param.requires_grad = False
    
    net = net_class(num_classes=num_classes)
    torch_device = torch.device(device)

    net.load_state_dict(state_dict)
    net.to(torch_device)
    
    test_AP = []
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=lr)
    criterion = YoloLoss(anchors=net.anchors)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        net.train()
        for _, (input, target) in tqdm(enumerate(loader), total=len(loader)):
            input.to(device)
            target.to(device)
            optimizer.zero_grad()

            #Yolo head is implemented in the loss for training, therefore yolo=False
            output = net(input, yolo=False)
            loss, _ = criterion(output, target)
            if writer:
                writer.add_scalar("Loss/train", loss)
            loss.backward()
            optimizer.step()
        
        test_precision = []
        test_recall = []
        net.eval()
        for _, (input, target) in tqdm(enumerate(loader_test), total=len(loader_test)):
            input.to(torch_device)
            target.to(torch_device)
            output = net(input, yolo=True)
            
            #The right threshold values can be adjusted for the target application
            output = filter_boxes(output, conf_threshold)
            output = nms(output, iou_threshold)

            for i in range(len(output)):
                precision, recall = precision_recall_levels(target[i], output[i])
                test_precision.append(precision)
                test_recall.append(recall) 

    test_AP.append(ap(test_precision, test_recall))
    sd = net.state_dict()
    return sd, test_AP


def train_qat(net_class: torch.nn.Module, num_classes: int, state_dict: Dict, loader: torch.utils.data.DataLoader, loader_test: torch.utils.data.DataLoader, writer=None, num_epochs: int=15, lr: float=0.001, freeze_q: bool=False, device: str="cpu"):
    net = net_class(num_classes=num_classes)
    torch_device = torch.device(device)

    net.load_state_dict(state_dict)
    net.to(torch_device)
    
    test_AP = []
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=lr)
    criterion = YoloLoss(anchors=net.anchors)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        net.train()
        for idx, (input, target) in tqdm(enumerate(loader), total=len(loader)):
            input.to(device)
            target.to(device)
            optimizer.zero_grad()

            if epoch>=3 and freeze_q:
                net.apply(torch.quantization.disable_observer)
                
            #Yolo head is implemented in the loss for training, therefore yolo=False
            output = net(input, yolo=False)
            loss, _ = criterion(output, target)
            if writer:
                writer.add_scalar("[QAT] Loss/train", loss)
            loss.backward()
            optimizer.step()
        
        test_precision, test_recall, _, _, _ = test(net, loader_test)
        test_AP.append(ap(test_precision, test_recall))

    return net, test_AP


def test(net_class: torch.nn.Module, num_classes: int, state_dict: Dict, loader_test: torch.utils.data.DataLoader, conf_threshold: float=0.1, iou_threshold: float=0.5, device: str="cpu"):
    net = net_class(num_classes=num_classes)
    torch_device = torch.device(device)

    net.load_state_dict(state_dict)
    net.to(torch_device)

    criterion = YoloLoss(anchors=net.anchors)
    
    test_precision = []
    test_recall = []
    losses = []
    inf_times = []
    running_loss = 0.0
    net.eval()
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(loader_test), total=len(loader_test)):
            input.to(torch_device)
            target.to(torch_device)

            t_start = time.time()
            output = net(input, yolo=True)
            t_end = time.time()
            
            #The right threshold values can be adjusted for the target application
            output = filter_boxes(output, conf_threshold)
            output = nms(output, iou_threshold)
            loss, _ = criterion(output, target)

            running_loss += loss.item()
            losses.append(loss.item())
            inf_times.append(t_end - t_start)

            for i in range(len(output)):
                precision, recall = precision_recall_levels(target[i], output[i])
                test_precision.append(precision)
                test_recall.append(recall) 

    avg_loss = running_loss / (i+1)

    return test_precision, test_recall, losses, avg_loss, inf_times
