Deploy a trained model to CMSSW
======

To deploy the trained model to CMSSW, the following three files from the model directory are needed:

 - `preprocessing.json`: this file contains the list of input variables and the input preprocessing information. It can be converted to a CMSSW python configuration fragment (e.g., [pfDeepBoostedJetPreprocessParams_cfi](https://github.com/cms-sw/cmssw/blob/master/RecoBTag/MXNet/python/pfDeepBoostedJetPreprocessParams_cfi.py)) using the [json2pset](json2pset.py) script.
 - `modelname-symbol.json`: this file describes the neural network architecture. 
    - [**Note**] For the mass decorrelated model, a couple of model json files exist, and the one named as `modelname-symbol-softmax.json` should be used.
 - `modelname-xxxx.params`: this file contains the trained parameters (i.e., weights) of the neural network. It needs to match the epoch number you intend to deploy.


[**Step 1**] Convert the `preprocessing.json` to a CMSSW python configuration fragment. This step needs to be done under a CMSSW release area.

```bash
cd CMSSW_X_Y_Z/src
cmsenv
# for the nominal tagger
python json2pset.py -i preprocessing.json -o pfDeepBoostedJetPreprocessParams_cfi.py -n pfDeepBoostedJetPreprocessParams
# for the mass decorrelated tagger
python json2pset.py -i preprocessing.json -o pfMassDecorrelatedDeepBoostedJetPreprocessParams_cfi.py -n pfMassDecorrelatedDeepBoostedJetPreprocessParams
```

The output `params.py` should replace either [pfDeepBoostedJetPreprocessParams_cfi](https://github.com/cms-sw/cmssw/blob/master/RecoBTag/MXNet/python/pfDeepBoostedJetPreprocessParams_cfi.py) or [pfMassDecorrelatedDeepBoostedJetPreprocessParams_cfi](https://github.com/cms-sw/cmssw/blob/master/RecoBTag/MXNet/python/pfMassDecorrelatedDeepBoostedJetPreprocessParams_cfi.py).


[**Step 2**] Gather the model files (`modelname-symbol.json` and `modelname-xxxx.params`) and integrate them into CMSSW. 
To have the models officially integrated in CMSSW, you need to add them to the [cms-data](https://github.com/cms-data/RecoBTag-Combined/tree/master/DeepBoostedJet/) repo via a pull request. 

[**Step 3**] Update the CMSSW configuration [pfDeepBoostedJet_cff.py](https://github.com/cms-sw/cmssw/blob/master/RecoBTag/MXNet/python/pfDeepBoostedJet_cff.py). Change the `model_path` and `param_path` to the model you intend to use. And verify that `pfDeepBoostedJetPreprocessParams` and `pfMassDecorrelatedDeepBoostedJetPreprocessParams` are consistent with the preprocessing parameters in `preprocessing.json`.