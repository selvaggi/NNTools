from __future__ import print_function

import os
import shutil
import argparse
from common import fit, data

import data.data_pfcands as dd

if __name__ == '__main__':
    # location of data
    train_val_fname = '/data/hqu/ntuples/20170619/pfcands/train_file_*.h5'
    test_fname = '/data/hqu/ntuples/20170619/pfcands/test_file_*.h5'
    example_fname = '/data/hqu/ntuples/20170619/pfcands/train_file_0.h5'
    example_val_fname = '/data/hqu/ntuples/20170619/pfcands/train_file_1.h5'


    # parse args
    parser = argparse.ArgumentParser(description="train pfcands",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    parser.set_defaults(
        # network
        network='resnet_simple',
        # config
        model_prefix='/data/hqu/training/mxnet/models/pfcands-20170619/resnet-simple/resnet',
        disp_batches=500,
        # data
        data_train=train_val_fname,
        train_val_split=0.8,
        data_test=test_fname,
        data_example=example_fname,
        data_names=','.join(dd.train_groups),
        label_names='softmax_label',
        weight_names='weight,class_weight',
        num_classes=-1,
        num_examples=-1,
        # train
        batch_size=1024,
        num_epochs=200,
        optimizer='adam',
        lr=1e-4,
        top_k=2,
        lr_step_epochs='20,40,60,80,120,160',
    )
    args = parser.parse_args()
    if args.dryrun:
        print('--DRY RUN--')
#         args.weight_names = ''
        args.data_train = example_fname
        args.data_val = example_val_fname
        args.num_examples = dd.nb_samples([example_fname])[0]
#         args.num_examples = dd.nb_wgt_samples([example_fname], args.weight_names)[0]

    if args.load_epoch:
        print('-' * 50)

#     n_train, n_val, n_test = dd.nb_wgt_samples([args.data_train, args.data_val, args.data_test], args.weight_names)
    n_train_val, n_test = dd.nb_samples([args.data_train, args.data_test])
    n_train = int(n_train_val * args.train_val_split)
    n_val = int(n_train_val * (1 - args.train_val_split))
    print(' --- Training sample size = %d, Validation sample size = %d, Test sample size = %d ---' % (n_train, n_val, n_test))
    args.num_examples = n_train
    args.num_classes = dd.nb_classes(example_fname)

    if args.predict:
        fit.predict(args, dd.load_data)
    else:
        # load network
        from importlib import import_module
        net = import_module('symbols.' + args.network)
        sym = net.get_symbol(**vars(args))

        save_dir = os.path.dirname(args.model_prefix)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        shutil.copy('symbols/%s.py' % args.network, save_dir)

        # train
        fit.fit(args, sym, dd.load_data)

