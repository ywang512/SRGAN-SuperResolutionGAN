# SRGAN-SuperResolutionGAN
SRGAN on satellite images implemented in PyTorch

## To Use

Change the parameters in `train.py` and use `python3 train.py` to run


## Outputs

Model weights, validation outputs and running statistics will be saved to `results/`. Under `*/val_predict/` folder, each image is one of the selected validation comparison between low-resolution (LR), high-resolution (HR) and super-resolution (SR) images from the left panel to the right.

`SR4` shows the result on ~500 64 x 64 LR images and ~500 256 x 256 HR images (factor of 4). These data could be found in `temp_data/`.

`SR4_15k` shows the result on ~15000 images with same settings as above.

`SR4_selected` shows the result on ~6000 selected images with same settings as above. That could be seen as a bench mark for now in our test.
