# SRGAN-SuperResolutionGAN
SRGAN on satellite images implemented in PyTorch

## To Use

Change the parameters in `train.py` and use `python3 train.py` to run


## Outputs

Model weights, validation outputs and running statistics will be saved to `results/`. Under `*/val_predict/` folder, each image is one of the selected validation comparison between low-resolution (LR), high-resolution (HR) and super-resolution (SR) images from the left panel to the right.

`SR4` shows the result on ~500 64 x 64 LR images and ~500 256 x 256 HR images (factor of 4). These data could be found in `temp_data/`.

`SR4_15k` shows the result on ~15000 images with same settings as above.


## Current issues and future steps

1. `d_loss`, `d_score` and `g_score` are not being correctly optimized. We might need to test and modify the loss function.

2. It seems working good on small dataset (see `results/SR4/` with only 500 images), but get completely trash on large dataset (see `results/SR4_15k/`). This might have something to do with the above loss function issues, but we are still not sure about that.

3. I got cuda `Out of Memory` errors when training with `batch_size > 1` (RTX 2060, 6GB). When running with `batch_size = 1`, each epoch takes ~2 min on `temp_data/`. That is approximately 250 images per minute.

4. (Outside of SRGAN) We also need a function to revert `chip_image_only()` to get the original size of them. This should be doable unless there is some intractable randomness in that function.
