Preprocessing module
======
Converts from root ntuples to PyTables files for DNN training.
 - variable transformation
 - weight calculation
 - ...


## Setup

### Option 1: Setup w/ LCG software stack (preferred)

For centos7 (e.g., LXPLUS):

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_96bpython3/x86_64-centos7-gcc9-opt/setup.sh
```



### Option 2: Setup w/ Miniconda (if LCG is not available)

Install miniconda if you don't have it:

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

Assume miniconda is installed at `$HOME/miniconda2` on LXPLUS.

```bash
# create a new conda environment
conda create -n prep python=2.7

# set up ROOT
mkdir -p $HOME/miniconda2/envs/prep/etc/conda/
cd $HOME/miniconda2/envs/prep/etc/conda/
mkdir activate.d  deactivate.d
cd activate.d
# create the env_vars.sh file to get ROOT environment
cat << EOF > env_vars.sh
#!/bin/sh
# $HOME/miniconda2/envs/prep/etc/conda/activate.d/env_vars.sh
echo "Source root environment..."
# ROOT
source /cvmfs/sft.cern.ch/lcg/external/gcc/4.9.1/x86_64-slc6/setup.sh
cd /cvmfs/sft.cern.ch/lcg/releases/ROOT/6.07.06-7096a/x86_64-slc6-gcc49-opt/
source bin/thisroot.sh
cd -
EOF

# activate the environment
# NOTE: should not run any cmsenv beforehand
source activate prep

# verify ROOT is activated
which root
(/cvmfs/sft.cern.ch/lcg/releases/ROOT/6.07.06-7096a/x86_64-slc6-gcc49-opt/bin/root)

# install the necessary python packages
pip install numpy numexpr pandas scikit-learn scipy tables
pip install root-numpy 

```
 
## Instructions

### Convert training/validation files

```bash
source activate prep
```
First activate the `prep` conda environment (if not yet done):

```bash
python runPreprocessing.py -n 50000 /eos/cms/store/path/to/input /eos/cms/store/path/to/output --data-format ak8_list --jobdir jobs -t condor
```
This will generate the condor submission script to convert the ntuples located at `/eos/cms/store/path/to/input` and output them to `/eos/cms/store/path/to/output`. Note that the input dir `/eos/cms/store/path/to/input` will be recursively searched and all files will be included, so **make sure you put testing samples outside this directory!**

 - It will first compute the metadata (i.e., the variable transformation, pT flattening weights, etc.) from the input root files. Note that this can take a long time so running with `tmux` or `screen` is recommended. The metadata will be saved in the output directory as `metadata.json` and can be re-used in the future (e.g., for converting the testing samples). 
 - The data format (e.g., what branches to include, reweighting method, whether to make jet images, etc.,) is specified by the `--data-format` option.  It should point to the python config file under `preprocessing/data_formats` (but without the .py suffix). 
 - `-t` option sets the job type (`condor` or `interactive`). For condor submission, the submit script is generated but you need to run the `condor_submit [your-submission-script]` command to actually submit the jobs.
 - `--jobdir` opetion sets the directory for job-related files (submission script, logs, etc.). Set different job dirs if you are runnning multiple jobs at the same time.
 - `-n` option sets the number of events for each output file. The default value, 50000, is good for the nominal data format (pfcand list, or image). 
 - If some of the jobs failed in condor, you can generate a resubmission script with only the failed jobs by invoking the `--resubmit` option:

```bash
python runPreprocessing.py -n 50000 /eos/cms/store/path/to/input /eos/cms/store/path/to/output --data-format ak8_list --jobdir jobs -t condor --resubmit
```

### Convert testing files (JMAR samples)

For evaluating the performance and make the ROC curves, the samples specified by JMAR (https://twiki.cern.ch/twiki/bin/view/CMS/JetMETHeavyResPaper) are used. Since a specific sample is used for each signal category (Top/W/Z/H), each sample needs to be converted separately.

**Note: the same metadata json file as used in converting the training file must be used for converting the testing files!**

Take the example of converting the top sample:

```bash
# create a separate directory for the top sample
mkdir -p /eos/cms/store/path/to/output/JMAR/Top
# copy the metadata file used for converting the training dataset
cp /eos/cms/store/path/to/output/metadata.json /eos/cms/store/path/to/output/JMAR/Top
# create the jobs
python runPreprocessing.py -n 50000 /eos/cms/store/path/to/JMAR_sample/JMAR/ZprimeToTT_M-3000_W-30_TuneCUETP8M1_13TeV-madgraphMLM-pythia8 /eos/cms/store/path/to/output/JMAR/Top --data-format ak8_list --jobdir jobs_Top --remake-filelist
```

 - Specify the path for the Top sample as the input path, and output the files to the `Top` directory.
 - `--remake-filelist` option is needed to update the input file list using the specified input directory (otherwise the input files in the metadata file will be used).
 - `--jobdir` opetion sets the directory for job-related files (submission script, logs, etc.). Set different job dirs if you are runnning multiple jobs at the same time.


### Reference preprocessing command

For nominal tagger training (94X, `20190326`)

```bash
python runPreprocessing.py -n 200000 /eos/cms/store/cmst3/group/deepjet/ak8/ntuples/94X/20190326_ak8_links /eos/cms/store/cmst3/group/deepjet/ak8/hqu/20190326_ak8/ak8puppi_parts --data-format "ak8_list" --jobdir ak8puppi_parts_20190326 &> ak8puppi_parts_list_20190326.log &
```

For decorrelated tagger training (94X, `20190326`)

```bash
# start with the same json to reuse the preprocessing parameters (median, lower/upper ranges)
# this would allow us to use the same testing files for both nominal tagger and the mass-decorrelated tagger
mkdir /eos/cms/store/cmst3/group/deepjet/ak8/hqu/20190326_ak8/ak8puppi_parts_ptmasswgt
cp /eos/cms/store/cmst3/group/deepjet/ak8/hqu/20190326_ak8/ak8puppi_parts/metadata.json /eos/cms/store/cmst3/group/deepjet/ak8/hqu/20190326_ak8/ak8puppi_parts_ptmasswgt

python runPreprocessing.py -n 200000 /eos/cms/store/cmst3/group/deepjet/ak8/ntuples/94X/20190326_ak8_links /eos/cms/store/cmst3/group/deepjet/ak8/hqu/20190326_ak8/ak8puppi_parts_ptmasswgt --data-format "ak8_list_ptmasswgt" --jobdir ak8puppi_parts_ptmasswgt_20190326 --remake-filelist --remake-weights &> ak8puppi_parts_list_ptmasswgt_20190326.log &
```
