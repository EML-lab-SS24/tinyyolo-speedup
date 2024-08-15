# ml-lab-ss24-challenge

This repos contains the code for the Embedded Machine Learning Lab Challenge. The focus of the lab was to speedup inference of TinyYolov2 by applying different steps: finetuning, layer fusion, pruning and other optimizations.

# TODOs

### Step 1
- [X] refactor training  
- [X] tune hyperparam (lr?)  
- [X] add early stopping to not overfit  
- [X] save model after each epoch  

### Step 2
- [X] Add quantized version of yolo  
- [X] add fusion and quantization with pytorch API (model 1: PTQ, model 2: QAT)  
- [X] test model 1 
- [X] train and test model 2  
- [X] add self-implemented fused conv_bn layer and quantized tinyyolo (model 3)  
- [X] train model 3  

Resources:  
- tutorial [quantization with pytorch](https://gist.github.com/martinferianc/d6090fffb4c95efed6f1152d5fde079d)  
- [pytorch quantization api](https://pytorch.org/docs/stable/quantization.html)   

### Step 3
- [X] Add pruning  

### Step 4
- [ ] export to ONNX
- [ ] test inference with ONNX

### Step 5
- [X] Add detection pipeline to camera loop  
- [X] Add framerate measurements  
- [X] Test each model for demo  

### Other tasks
- [ ] add tensorboard logging for every step
- [ ] Add visuals for different models  
- see [this graphs](docs/images/visuals_ideas_deep_compression_paper.png) from [this paper](https://arxiv.org/pdf/1510.00149)  
- [ ] compare roc curves of different models (one plot with all curves)  
- [ ] compare size of models (stae_dicts) after each step (histogram)  
- [ ] implement test for inference time improvement (see last cell in quantization notebook)  

### Extensions
- [ ] Other pruning method: https://github.com/NVlabs/Taylor_pruning
- [ ] ONNX Graph optimization: https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
- [ ] Up-to-date Jetson Docker container https://github.com/dusty-nv/jetson-containers
