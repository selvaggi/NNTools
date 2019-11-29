from __future__ import print_function

import os
import shutil
import argparse
from common import fit, data
from importlib import import_module

if __name__ == '__main__':
    # location of data
    train_val_fname = '/data/hqu/ntuples/20170717/pfcands_minor_labels/train_file_*.h5'
    test_fname = '/data/hqu/ntuples/20170717/pfcands_minor_labels/testing/train_file_*.h5'
    example_fname = '/data/hqu/ntuples/20170717/pfcands_minor_labels/train_file_0.h5'

    # parse args
    parser = argparse.ArgumentParser(description="train pfcands",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    parser.set_defaults(
        # network
        network='resnet_simple',
        # config
        model_prefix='/data/hqu/training/mxnet/models/pfcands_minor_labels-20170717/resnet-simple/resnet',
        disp_batches=500,
        # data
        data_config='data_pfcands',
        data_train=train_val_fname,
        train_val_split=0.8,
        data_test=test_fname,
        data_example=example_fname,
        data_names=None,
        label_names='softmax_label',
        weight_names='weight,class_weight',
        num_examples=-1,
        # train
        batch_size=1024,
        num_epochs=200,
        optimizer='adam',
        lr=1e-3,
        top_k=2,
        lr_step_epochs='10,20,30,50',
    )
    args = parser.parse_args()
    # load data config
    dd = import_module('data.' + args.data_config)
    args.data_names = ','.join(dd.train_groups)

    if args.dryrun:
        print('--DRY RUN--')
#         args.weight_names = ''
        args.data_train = example_fname
        args.train_val_split = 0.5
#         args.num_examples = dd.nb_wgt_samples([example_fname], args.weight_names)[0]

    if args.load_epoch:
        print('-' * 50)

#     n_train, n_val, n_test = dd.nb_wgt_samples([args.data_train, args.data_val, args.data_test], args.weight_names)
    n_train_val, n_test = dd.nb_samples([args.data_train, args.data_test])
    n_train = int(n_train_val * args.train_val_split)
    n_val = int(n_train_val * (1 - args.train_val_split))
    print(' --- Training sample size = %d, Validation sample size = %d, Test sample size = %d ---' % (n_train, n_val, n_test))
    if args.num_examples < 0:
        args.num_examples = n_train

    if args.predict:
        # load network
        sym = import_module('symbols.' + args.network)

        fit.predict(args, sym, dd.load_data)
    else:
        # load network
        sym = import_module('symbols.' + args.network)

        save_dir = os.path.dirname(args.model_prefix)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        shutil.copy('symbols/%s.py' % args.network, save_dir)

        # train
        fit.fit(args, sym, dd.load_data)

