Training module
======
Load the converted PyTables files and train DNNs with MXNet.

## Setup

#### Install miniconda if you don't have it:

```bash
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda2-latest-Linux-x86_64.sh
# Follow the insturctions to finish the installation
```

Verify the installation is successful by running `conda info`.

If you cannot run `conda` command, check if the you added the conda path to your `PATH` variable in your bashrc/zshrc file, e.g., 

```bash
export PATH="$HOME/miniconda2/bin:$PATH"
```

#### Set up the environment for training

The following instruction is only for training with Nvidia GPU. CUDA 8.0 and cuDNN (>=5) is required.

```bash
# create a new conda environment
conda create -n mxnet python=2.7

# set up ROOT
mkdir -p $HOME/miniconda2/envs/mxnet/etc/conda/
cd $HOME/miniconda2/envs/mxnet/etc/conda/
mkdir activate.d  deactivate.d
cd activate.d
# create the env_vars.sh file with the following content:
# change the ROOT path according to your environment
# ------
#!/bin/sh
# $HOME/miniconda2/envs/mxnet/etc/conda/activate.d/env_vars.sh
echo "Source root environment..."
# ROOT
source /cvmfs/sft.cern.ch/lcg/external/gcc/4.9.1/x86_64-slc6/setup.sh
cd /cvmfs/sft.cern.ch/lcg/releases/ROOT/6.07.06-7096a/x86_64-slc6-gcc49-opt/
source bin/thisroot.sh
cd -
# ------

# activate the environment
source activate mxnet

# install the necessary python packages
pip install numpy numexpr pandas scikit-learn scipy tables
pip install root-numpy 
pip install mxnet-cu80
```
 
## Instructions

#### Start a new training

```bash
python train_pfcands_simple.py --network resnet_simple --model-prefix /path/to/model/checkpoints/model-name-without-suffix --batch-size 512 --optimizer adam --lr 0.001 --lr-step-epochs "10,20,30,50" --num-epochs 80 --data-train '/path/to/data/train_file_*.h5' --dataloader-nworkers 2 --dataloader-qsize 256 --gpus 0 &> /path/to/logfile.log &
```

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
python train_pfcands_simple.py --network resnet_simple --model-prefix /path/to/model/checkpoints/model-name-without-suffix --batch-size 512 --optimizer adam --lr 0.001 --lr-step-epochs "10,20,30,50" --num-epochs 80 --data-train '/path/to/data/train_file_*.h5' --dataloader-nworkers 2 --dataloader-qsize 256 --gpus 0 --load-epoch 20 &>> /path/to/logfile.log &
```

 - Use `--load-epoch` option to load the checkpoint and resume the training (e.g., `--load-epoch 20` will resume the training from the Epoch 20).
 - `&>>` allows you to append to the log file instead of overwriting it.
 
#### Run prediction with trained model

```bash
python train_pfcands_simple.py --predict --load-epoch 60 --network resnet_simple --model-prefix /path/to/model/checkpoints/model-name-without-suffix  --data-train '/path/to/data/train_file_*.h5' --batch-size 32 --dataloader-nworkers 2 --dataloader-qsize 50 --gpus 0 --data-test '/path/to/test-data/JMAR/Top/train_file_*.h5' --predict-output /path/to/output/mx-pred_Top.h5
```
 - `--predict`: run prediction instead of training.
 - `--load-epoch`: load the model parameter from which epoch.
 - `--batch-size 32`: a smaller batch size is preferred in prediction mode to avoid losing events.
 - `--data-test`: path for the testing files; support Unix style pathname pattern expansion (i.e., `*` and `?`) using `glob` in python, but make sure you wrap it with single quote (`'`).
 - `--predict-output`: output file. Both PyTables (`.h5`) and root file will be created.

