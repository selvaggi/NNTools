Training module
======
Load the converted PyTables files and train DNNs with MXNet.

## Setup

#### Install miniconda if you don't have it:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the insturctions to finish the installation
```

Verify the installation is successful by running `conda info`.

If you cannot run `conda` command, check if the you added the conda path to your `PATH` variable in your bashrc/zshrc file, e.g., 

```bash
export PATH="$HOME/miniconda3/bin:$PATH"
```

#### Set up the environment for training

The following instruction is only for training with Nvidia GPU. CUDA 8.0 and cuDNN (>=5) is required.

```bash
# create a new conda environment
conda create -n mxnet python=3.7

# set up ROOT
# (below assumes centos7, for other systems please modify the ROOT installation path accordingly)
mkdir -p $HOME/miniconda3/envs/mxnet/etc/conda/
cd $HOME/miniconda3/envs/mxnet/etc/conda/
mkdir activate.d  deactivate.d
cd activate.d
# create the env_vars.sh file to get ROOT environment
cat << EOF > env_vars.sh
#!/bin/sh
# $HOME/miniconda3/envs/prep/etc/conda/activate.d/env_vars.sh
echo "Source root environment..."
# ROOT
source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.14.06/x86_64-centos7-gcc48-opt/bin/thisroot.sh
EOF

# activate the environment
source activate mxnet

# install the necessary python packages
conda install -c anaconda hdf5
pip install numpy numexpr pandas scikit-learn scipy tables matplotlib
pip install root-numpy

# install mxnet -- this depends on the CUDA version (the current recommendation is CUDA 10.1)
pip install mxnet-cu101

# for other CUDA versions, please check https://mxnet.incubator.apache.org/install/
```
 
## Instructions

#### Start a new training

```bash
python train_pfcands_simple.py --data-config data_ak8_parts_sv --network resnet_simple --model-prefix /path/to/model/checkpoints/model-name-without-suffix --batch-size 512 --optimizer adam --lr 0.001 --lr-step-epochs "10,20,30,50" --num-epochs 80 --data-train '/path/to/data/train_file_*.h5' --dataloader-nworkers 2 --dataloader-qsize 32 --gpus 0 &> /path/to/logfile.log &
```

 - `--data-config`: which configuration of the inputs to use. `data_ak8_parts_sv` -> `data/data_ak8_parts_sv.py`.
 - `--network`: the DNN model to use. `resnet_simple` -> `symbols/resnet_simple.py`.
 - `--model-prefix`: path for saving training checkpoints at the end of each epoch. The saved model can be used restarting a interrupted training, as well as running predictions to evaluate the performance.
 - `--batch-size`: minibatch size for training. Adjust this according the model complexity to fit the GPU memory. This can also be tuned as a hyperparameter.
 - `--optimizer`: training optimizer. Currently support `adam` and `sgd`.
 - `--lr`: learning rate.
 - `--lr-step-epochs`: the epochs to reduce the lr by `--lr-factor` (defaults to 0.1), e.g., "10,20,30,50" means the 10th, 20th, 30th, and 50th epoch
 - `--num-epochs`: max number of epochs to run
 - `--data-train`: path for the training files; support Unix style pathname pattern expansion (i.e., `*` and `?`) using `glob` in python, but make sure you wrap it with single quote (`'`).
 - `--dataloader-nworkers`: number of parallel threads for loading the dataset.
 - `--dataloader-qsize`: queue size of the dataloader (adjust according to the RAM size and `--dataloader-nworkers`).
 - `--gpus`: set which GPU to use. Multiple GPUs can be specified as a comma seperated string, e.g., `"0,1,2,3"`. Set to an empty string `""` if you want to use CPU.
 - More options can be found by running `python train_pfcands_simple.py -h` or checking the source code.
 - `&> /path/to/logfile.log &` will redirect both stdout/stderr to the file `/path/to/logfile.log`, and the training `&` will run this process in the background. You can view the log file with `less` (e.g., type `F` to follow the tail of the file).
 
#### Resume an interrupted training

```bash
python train_pfcands_simple.py --data-config data_ak8_parts_sv --network resnet_simple --model-prefix /path/to/model/checkpoints/model-name-without-suffix --batch-size 512 --optimizer adam --lr 0.001 --lr-step-epochs "10,20,30,50" --num-epochs 80 --data-train '/path/to/data/train_file_*.h5' --dataloader-nworkers 2 --dataloader-qsize 32 --gpus 0 --load-epoch 20 &>> /path/to/logfile.log &
```

 - Use `--load-epoch` option to load the checkpoint and resume the training (e.g., `--load-epoch 20` will resume the training from the Epoch 20).
 - `&>>` allows you to append to the log file instead of overwriting it.
 - Note that although this is possible, it is not recommended in general as some optimzers have weight decay which depends on the number of epoch.
 
#### Run prediction with trained model

```bash
python train_pfcands_simple.py --data-config data_ak8_parts_sv --network resnet_simple --model-prefix /path/to/model/checkpoints/model-name-without-suffix --load-epoch 60 --batch-size 32 --data-train '/path/to/data/train_file_*.h5' --dataloader-nworkers 2 --dataloader-qsize 32 --gpus 0 --predict --data-test '/path/to/test-data/JMAR/Top/train_file_*.h5' --predict-output /path/to/output/mx-pred_Top.h5
```
 - `--predict`: run in prediction mode instead of training.
 - `--load-epoch`: load the model parameter from which epoch (e.g., `--load-epoch 5` will load `model-0005.params`).
 - `--batch-size 32`: a smaller batch size is preferred in prediction mode to avoid losing events.
 - `--data-test`: path for the testing files; support Unix style pathname pattern expansion (i.e., `*` and `?`) using `glob` in python, but make sure you wrap it with single quote (`'`).
 - `--predict-output`: output file. Both PyTables (`.h5`) and root file will be created.

### Reference training/prediction command

Nominal version (94X, `V1`)

Training:

```bash
python train_pfcands_simple.py --data-config data_ak8_pfcand_sv --network sym_ak8_pfcand_sv_resnet_v1 --model-prefix /data/hqu/training/mxnet/models/20190326_ak8_classrewgt/pfcand_sv_resnet_v1/resnet --batch-size 1024 --optimizer adam --lr 0.001 --lr-step-epochs "15,30,40" --num-epochs 50 --data-train '/data/hqu/ntuples/20190326_ak8/ak8puppi_parts_classrewgt/train_file_*.h5' --train-val-split 0.9 --dataloader-nworkers 3 --dataloader-qsize 48 --disp-batches 1000 --gpus 0 &> logs/train_ak8puppi_20190326_classrewgt_pfcand_sv_ref_resnet_v1.log &
```

Prediction:

```bash
python train_pfcands_simple.py --data-config data_ak8_pfcand_sv --network sym_ak8_pfcand_sv_resnet_v1 --model-prefix /data/hqu/training/mxnet/models/20190326_ak8_classrewgt/pfcand_sv_resnet_v1/resnet --load-epoch 39 --batch-size 128 --data-train '/data/hqu/ntuples/20190326_ak8/ak8puppi_parts_classrewgt/train_file_*.h5' --data-test '/data/hqu/ntuples/20190326_ak8/test_samples/JMAR/QCD/train_file_*.h5' --predict-output /data/hqu/training/mxnet/predict/20190326_ak8_classrewgt/pfcand_sv_resnet_v1/epoch39/JMAR/mx-pred_QCD.h5 --dataloader-nworkers 2 --dataloader-qsize 16 --gpus 0 --predict --predict-all &> logs/preds/pred_ak8puppi_20190326_classrewgt_pfcand_sv_ref_resnet_simple_epoch39.log &
```


Decorrelated version (94X, `V1`)

Training:

```bash
python train_features_adv.py \
 --data-config data_ak8_adv_pfcand_sv \
 --network block_ak8_adv_resnet_features_r_3x256_pfcand_sv_dropout \
 --model-prefix /data/hqu/training/mxnet/models/20190326_ak8_adv/pfcand_sv_resnet_features_r_3x256_dropout_mass30to250_22bins_advwgt5_advfreq10_lr_1e-2_decay0p1_30_60_90_advlr_1e-4_batch8k/resnet \
 --data-train '/data/hqu/ntuples/20190326_ak8/ak8puppi_parts_ptmasswgt/train_file_*.h5' \
 --dataloader-weight-scale 1 --dataloader-max-resample 100 --dataloader-nworkers 2 --dataloader-qsize 16 \
 --batch-size 8192 --num-epochs 120 --train-val-split 0.9 \
 --optimizer adam --lr 1e-2 --lr-factor 0.1 --lr-step-epochs "30,60,90" \
 --adv-lr 1e-4 --adv-lr-factor 0.1 --adv-lr-step-epochs "1000" \
 --adv-lambda 5 --adv-mass-min 30 --adv-mass-max 250 --adv-mass-nbins 22 --adv-train-freq 10 \
 --gpus 0 --disp-batches 100 \
 &> logs/dev-adv-ak8puppi_20190326_ptmasswgt_pfcand_sv_resnet_features_r_3x256_dropout_mass30to250_22bins_advwgt5_advfreq10_lr_1e-2_decay0p1_30_60_90_advlr_1e-4_batch8k.log &
```

Prediction:

```bash
python train_features_adv.py \
 --data-config data_ak8_adv_pfcand_sv \                                     
 --network block_ak8_adv_resnet_features_r_3x256_pfcand_sv_dropout \
 --model-prefix /data/hqu/training/mxnet/models/20190326_ak8_adv/pfcand_sv_resnet_features_r_3x256_dropout_mass30to250_22bins_advwgt5_advfreq10_lr_1e-2_decay0p1_30_60_90_advlr_1e-4_batch8k/resnet \
 --data-train '/data/hqu/ntuples/20190326_ak8/ak8puppi_parts_ptmasswgt/train_file_*.h5' \
 --dataloader-nworkers 2 --dataloader-qsize 16 \
 --batch-size 128 --data-test '/data/hqu/ntuples/20190326_ak8/test_samples/JMAR/QCD/train_file_*.h5' \
 --load-epoch 50 --predict-output /data/hqu/training/mxnet/predict/20190326_ak8_adv/pfcand_sv_resnet_features_r_3x256_dropout_mass30to250_22bins_advwgt5_advfreq10_lr_1e-2_decay0p1_30_60_90_advlr_1e-4_batch8k/JMAR/mx-pred_QCD.h5 \
 --predict --predict-all --predict-epochs "70,99,119" \
 --gpus 1 --disp-batches 100 \
 &> logs/preds/preds-adv-ak8puppi_20190326_ptmasswgt_pfcand_sv_resnet_features_r_3x256_dropout_mass30to250_22bins_advwgt5_advfreq10_lr_1e-2_decay0p1_30_60_90_advlr_1e-4_batch8k_epoch_70_99_119.log &
```
