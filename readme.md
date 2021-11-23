# Coherent Grid Image Processing with Adaptive Normalization Layers

## Setup

Install the dependencies with pip

```
pip install -r requirements.txt

```

This code also requires the Synchronized-BatchNorm-PyTorch:

```
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
rm -rf Synchronized-BatchNorm-PyTorch
cd ../../
```

## Datasets

### Mit 5K

Mit 5K dataset can be obtained at [**MIT-Adobe FiveK**](https://data.csail.mit.edu/graphics/fivek/).

For training/testing, data should have the following structure

```
# Training files limited to i.e. 512
mit-5k/<size>/original/
mit-5k/<size>/C/

# Full resolution testing files
mit-5k/evaluation/original/
mit-5k/evaluation/C/

# Limited resolution testing files
mit-5k/evaluation-1024/original/
mit-5k/evaluation-1024/C/

```

### Facades CMP

For the Facades CMP dataset, run:

```
cd data 
sh facadesHR_download_and_extract.sh
```

## Training

Use `launch-train-mit5k.sh` and `launch-train-facades.sh` for training the default models.


## Testing 

Use `launch-test-mit5k.sh` and `launch-test-facades.sh` for training the default models.


## Testing pre-trained models

### Mit 5K

Download the [mit 5k model](https://drive.google.com/file/d/1h2GbU-bQuXOZDejM7Zq-wiwJhsR9LxYa/view) and place it in `./checkpoints/mit5k-default/best-psnr_net_G.pth` and run `launch-test-mit5k.sh`.

### Facades CMP

Download the [facades model](https://drive.google.com/file/d/1TZSOkmGVzcGfZMelL6ChxzE-wzdTGJ3y/view) and place it in `./checkpoints/facades-default/best-psnr_net_G.pth` and run `launch-test-facades.sh`.


# Acknowledgments

The code is inspired by [ASAPNet](https://github.com/tamarott/ASAPNet), [SPADE](https://github.com/NVlabs/SPADE) & [
Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)