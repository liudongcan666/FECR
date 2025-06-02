# FECR

A PyTorch-based implementation for the **FECR** model for person re-identification tasks.

## Getting Started

This section describes how to prepare your environment and run training or evaluation.

### 🏋️‍♂️ Training

1. **Prepare Dataset**

   Create a folder named `data/`, and put the datasets inside. Modify the corresponding dataset paths in the configuration files to match your setup.

2. **Download Pretrained Weights**

   Download the pretrained weights used in the paper (e.g., ViT pretrained on ImageNet). Place them in the appropriate directory and update the configuration files with their paths.

3. **Start Training**

   Run the following commands to train the model. You will need a GPU with at least **16GB memory**.

   ```bash
   # Training on Occluded-Duke
   python train.py --config_file configs/OCC_Duke/vit_transreid_stride.yml MODEL.DEVICE_ID "('2')"

   # Training on Occluded-ReID
   python train.py --config_file configs/OCC_ReID/vit_transreid_stride.yml MODEL.DEVICE_ID "('2')"

   # Training on Market-1501
   python train.py --config_file configs/Market/vit_transreid_stride.yml MODEL.DEVICE_ID "('2')"

   # Training on DukeMTMC-reID
   python train.py --config_file configs/DukeMTMC/vit_transreid_stride.yml MODEL.DEVICE_ID "('2')"
