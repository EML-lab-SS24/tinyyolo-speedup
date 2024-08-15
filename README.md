# ml-lab-ss24-challenge

# TODOs

### Step 1
- [ ] refactor training  
- [ ] tune hyperparam (lr?)  
- [ ] add early stopping to not overfit  
- [ ] save model after each epoch  

### Step 2
- [X] Add quantized version of yolo  
- [X] add fusion and quantization with pytorch API (model 1: PTQ, model 2: QAT)  
- [X] test model 1 
- [X] train and test model 2  
- [ ] add self-implemented fused conv_bn layer and quantized tinyyolo (model 3)  
- [ ] train model 3  

Resources:  
- tutorial [quantization with pytorch](https://gist.github.com/martinferianc/d6090fffb4c95efed6f1152d5fde079d)  
- [pytorch quantization api](https://pytorch.org/docs/stable/quantization.html)   

### Step 3
- [ ] Add pruning to quantization  

### Step 4
- [ ] export to ONNX
- [ ] test inference with ONNX

### Step 5
- [ ] Add detection pipeline to camera loop  
- [ ] Add framerate measurements  
- [ ] Test each model for demo  

### Other tasks
- [ ] add tensorboard logging for every step
- [ ] Add visuals for different models  
- see [this graphs](docs/images/visuals_ideas_deep_compression_paper.png) from [this paper](https://arxiv.org/pdf/1510.00149)  
- [ ] compare mAP of different modes (histogram)  
- [ ] compare roc curves of different models (one plot with all curves)  
- [ ] compare size of models (stae_dicts) after each step (histogram)  
- [ ] implement test for inference time improvement (see last cell in quantization notebook)  
- [ ] Add mAP (50) calculation (see utils?)  


# Training results
finetuning
- last 2 layers: converges after 15 epochs
- last layer: converges after 75 epochs

fusion:
- loosing accuracy after fusion
- ?did not loose accuracy after fusion on finetuned 7 frozen layers, training takes longer
