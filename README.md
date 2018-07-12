# NNTools

Lightweight framework for streamlined maching learning development for high energy physics, including
 - data conversion from ROOT trees to [PyTables](https://www.pytables.org/)
 - data preprocessing, including input transformation/standardization, dynamic sample reweighting, etc.
 - neural network training with [Apache MXNet](https://mxnet.incubator.apache.org/), checkpoint save, model exportation, etc.

This framework has been used for the development of the [DeepAK8 tagger](https://twiki.cern.ch/twiki/bin/viewauth/CMS/DeepAKXTagging) in CMS, more details of which can be found in [CMS-DP-2017-049](https://cds.cern.ch/record/2295725?ln=en) and [NIPS 2017 workshop paper](https://dl4physicalsciences.github.io/files/nips_dlps_2017_10.pdf).

More details about how to use this framework can be found in the README files of the [preprocessing](preprocessing) module and the [training](training) module.
