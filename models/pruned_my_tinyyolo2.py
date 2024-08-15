from models.my_tinyyolov2 import MyTinyYoloV2
import torch
import torch.nn as nn


class PrunedMyTinyYoloV2(MyTinyYoloV2):
    def __init__(self, num_classes=20, anchors=[]):
        super().__init__(num_classes, anchors)
                
        self._register_load_state_dict_pre_hook(self._sd_hook)

    def _sd_hook(self, state_dict, prefix, *_):
        for key in state_dict:
            if 'conv' in key and 'weight' in key:
                n = int(key.split('conv')[1].split('.weight')[0])
            else:
                continue

            dim_in = state_dict[f'conv{n}.weight'].shape[1]
            dim_out = state_dict[f'conv{n}.weight'].shape[0]
            
            if n == 9:
                self.conv9 = nn.Conv2d(dim_in, dim_out, 1, 1, padding=0)
                continue

            conv = nn.Conv2d(dim_in, dim_out, 3, 1, padding=1)
            if n == 1: self.conv1 = conv
            elif n == 2: self.conv2 = conv
            elif n == 3: self.conv3 = conv
            elif n == 4: self.conv4 = conv
            elif n == 5: self.conv5 = conv
            elif n == 6: self.conv6 = conv
            elif n == 7: self.conv7 = conv
            elif n == 8: self.conv8 = conv
        pass