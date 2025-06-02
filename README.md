# FECR

## Getting Started


###  Training

1. Prepare data
   Place the datasets into the `data/` directory and update the corresponding dataset paths in the configuration files.

3. Download pretrained weights
   Download the pretrained weights and modify the configuration files to point to the correct local paths.

4. Start training
   Run the following commands to train the model. A GPU with at least 16GB memory is recommended.
```
   # Training on Occluded-Duke
   python train.py --config_file configs/OCC_Duke/vit_transreid_stride.yml MODEL.DEVICE_ID "('2')"

   # Training on Occluded-ReID
   python train.py --config_file configs/OCC_ReID/vit_transreid_stride.yml MODEL.DEVICE_ID "('2')"

   # Training on Market-1501
   python train.py --config_file configs/Market/vit_transreid_stride.yml MODEL.DEVICE_ID "('2')"

   # Training on DukeMTMC-reID
   python train.py --config_file configs/DukeMTMC/vit_transreid_stride.yml MODEL.DEVICE_ID "('2')"
```
###  Testing

1. Download checkpoints
   Link: 'https://pan.baidu.com/s/1Y79ycKRRF-7EDyNZ3evA7w'
   Code: '445v'
   Extract the downloaded archive after completion.

2. Run testing
   Place the extracted checkpoints in the correct location, then run the `test.py` script to evaluate the model.

###  Requirements

Please refer to `requirements.txt` for the list of dependencies.

