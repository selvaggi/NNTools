from __future__ import print_function

from common.data import DataFormat, DataLoader
import glob
import os
import logging

label_var = 'label'
train_groups = ['img']
train_vars = {}
train_vars['img'] = (
    'img',
    )
obs_vars = [
    'orig_fj_labelJMAR', 'orig_fjJMAR_gen_pt', 'orig_fjJMAR_gen_eta', 'orig_fjJMAR_gen_pdgid',
    'orig_fj_labelLegacy',
    'orig_fj_label',
    'orig_fj_isQCD', 'orig_fj_isTop', 'orig_fj_isW', 'orig_fj_isZ', 'orig_fj_isH',

    'orig_fj_pt',
    'orig_fj_eta',
    'orig_fj_phi',
    'orig_fj_mass',
    'orig_fj_sdmass',
    'orig_fj_n_sdsubjets',
    'orig_fj_doubleb',
    'orig_pfCombinedInclusiveSecondaryVertexV2BJetTags',
    'orig_fj_tau21',
    'orig_fj_tau32',
    'orig_npv',
    'orig_n_pfcands',
    'orig_n_tracks',
    'orig_n_sv',
    'orig_fjPuppi_tau21', 'orig_fjPuppi_tau32', 'orig_fjPuppi_corrsdmass',
    'orig_fj_sdsj1_csv', 'orig_fj_sdsj2_csv',
    ]

def load_data(args):

    train_val_filelist = glob.glob(args.data_train)
    n_train = int(args.train_val_split * len(train_val_filelist))

    wgtvar = args.weight_names
    if wgtvar == '': wgtvar = None

    d = DataFormat(train_groups, train_vars, label_var, wgtvar, obs_vars, filename=train_val_filelist[0])

    logging.info('Using the following variables:\n' +
                 '\n'.join([v_group + '\n\t' + str(train_vars[v_group]) for v_group in train_groups ]))
    logging.info('Using weight\n' + wgtvar)

    orig_metadata = os.path.join(os.path.dirname(train_val_filelist[0]), 'metadata.json')
    output_metadata = os.path.join(os.path.dirname(args.model_prefix), 'preprocessing.json')

    if args.predict:
        test_filelist = glob.glob(args.data_test)
        test = DataLoader(test_filelist, d, batch_size=args.batch_size, predict_mode=True, shuffle=False, args=args)
        return test
    else:
        train = DataLoader(train_val_filelist[:n_train], d, batch_size=args.batch_size, args=args)
        val = DataLoader(train_val_filelist[n_train:], d, batch_size=args.batch_size, args=args)
        if not os.path.exists(output_metadata):
            train_shapes = {}
            for k, v in train.provide_data:
                train_shapes[k] = (1,) + v[1:]
            dump_input_metadata(orig_metadata, groups=train_groups, shapes=train_shapes,
                                var_names=train_vars, output=output_metadata)
        return (train, val)

def nb_samples(files):
    nevts = []
    for f in files:
        filelist = glob.glob(f)
        nevts.append(sum([DataFormat.nevts(filename, label_var) for filename in filelist]))
    return tuple(nevts)

def nb_classes(filename):
    return DataFormat.num_classes(filename, label_var)

def nb_wgt_samples(files, weight_names):
    if not weight_names:
        return nb_samples(files)

    nevts = []
    for f in files:
        filelist = glob.glob(f)
        nevts.append(int(sum([DataFormat.nwgtsum(filename, weight_names) for filename in filelist])))
    return tuple(nevts)

def dump_input_metadata(orig_metadata, groups, shapes, var_names, output='inputs.json'):
    out = {}
    import json
    with open(orig_metadata) as f:
        orig = json.load(f)
    out['var_info'] = orig['branches_info']
    out['input_names'] = groups
    out['input_shapes'] = shapes
    out['var_names'] = var_names
#     print(out)
    with open(output, 'w') as f:
        json.dump(out, f, indent=2, sort_keys=True)
    logging.info('Output json file to %s' % output)
