from __future__ import print_function

from common.data import DataFormat, DataLoader
import glob
import os
import logging

label_var = 'label'
extra_label_vars = ['orig_fj_sdmass_fromsubjets', 'orig_fj_pt']  # [mass, pt]
train_groups = ['pfcand', 'sv']
train_vars = {}
train_vars['pfcand'] = (
    # basic kinematics
    'pfcand_pt_log',
    'pfcand_ptrel_log',
    'pfcand_erel_log',
    'pfcand_phirel',
    'pfcand_etarel',
    'pfcand_deltaR',
    'pfcand_abseta',
    'pfcand_puppiw',

    'pfcand_drminsv',
    'pfcand_drsubjet1',
    'pfcand_drsubjet2',

    'pfcand_charge',

    # track quality
    'pfcand_VTX_ass',
#     'pfcand_fromPV',
    'pfcand_lostInnerHits',
#    'pfcand_trackHighPurity',
    'pfcand_normchi2',
    'pfcand_quality',

    # impact parameters
    'pfcand_dz',
    'pfcand_dzsig',
    'pfcand_dxy',
    'pfcand_dxysig',

    # track covariance
    'pfcand_dptdpt',
    'pfcand_detadeta',
    'pfcand_dphidphi',
    'pfcand_dxydxy',
    'pfcand_dzdz',
    'pfcand_dxydz',
    'pfcand_dphidxy',
    'pfcand_dlambdadz',

    # track btag info
#     'pfcand_btagMomentum',
#     'pfcand_btagEta',
    'pfcand_btagEtaRel',
#     'pfcand_btagPtRel',
#     'pfcand_btagPPar',
#     'pfcand_btagDeltaR',
    'pfcand_btagPtRatio',
    'pfcand_btagPParRatio',
    'pfcand_btagSip2dVal',
    'pfcand_btagSip2dSig',
    'pfcand_btagSip3dVal',
    'pfcand_btagSip3dSig',
    'pfcand_btagJetDistVal',

    )

train_vars['sv'] = (
    'sv_pt_log',
    'sv_ptrel_log',
    'sv_erel_log',
    'sv_phirel',
    'sv_etarel',
    'sv_deltaR',
    'sv_abseta',
    'sv_mass',

    'sv_ntracks',
    'sv_normchi2',
    'sv_dxy',
    'sv_dxysig',
    'sv_d3d',
    'sv_d3dsig',
    'sv_costhetasvpv',
    )

obs_vars = [
    'orig_event_no',
    'orig_fj_labelJMAR', 'orig_fjJMAR_gen_pt', 'orig_fjJMAR_gen_eta', 'orig_fjJMAR_gen_pdgid',
    'orig_fj_isQCD', 'orig_fj_isTop', 'orig_fj_isW', 'orig_fj_isZ', 'orig_fj_isH',

    'orig_fj_pt',
    'orig_fj_eta',
    'orig_fj_phi',
    'orig_fj_mass',
    'orig_fj_sdmass',
    'orig_fj_sdmass_fromsubjets',
    'orig_fj_corrsdmass',
    'orig_fj_n_sdsubjets',
    'orig_fj_nbHadrons',
    'orig_fj_ncHadrons',
    'orig_fj_doubleb',
    'orig_fj_tau21',
    'orig_fj_tau32',
    'orig_npv',
    'orig_n_pfcands',
    'orig_n_sv',

    'orig_pfDeepBoostedJetTags_probTbcq',
    'orig_pfDeepBoostedJetTags_probTbqq',
    'orig_pfDeepBoostedJetTags_probTbc',
    'orig_pfDeepBoostedJetTags_probTbq',
    'orig_pfDeepBoostedJetTags_probWcq',
    'orig_pfDeepBoostedJetTags_probWqq',
    'orig_pfDeepBoostedJetTags_probZbb',
    'orig_pfDeepBoostedJetTags_probZcc',
    'orig_pfDeepBoostedJetTags_probZqq',
    'orig_pfDeepBoostedJetTags_probHbb',
    'orig_pfDeepBoostedJetTags_probHcc',
    'orig_pfDeepBoostedJetTags_probHqqqq',
    'orig_pfDeepBoostedJetTags_probQCDbb',
    'orig_pfDeepBoostedJetTags_probQCDcc',
    'orig_pfDeepBoostedJetTags_probQCDb',
    'orig_pfDeepBoostedJetTags_probQCDc',
    'orig_pfDeepBoostedJetTags_probQCDothers',
    'orig_pfDeepBoostedDiscriminatorsJetTags_TvsQCD',
    'orig_pfDeepBoostedDiscriminatorsJetTags_WvsQCD',
    'orig_pfDeepBoostedDiscriminatorsJetTags_ZvsQCD',
    'orig_pfDeepBoostedDiscriminatorsJetTags_ZbbvsQCD',
    'orig_pfDeepBoostedDiscriminatorsJetTags_HbbvsQCD',
    'orig_pfDeepBoostedDiscriminatorsJetTags_H4qvsQCD',

    'orig_pfMassDecorrelatedDeepBoostedJetTags_probTbcq',
    'orig_pfMassDecorrelatedDeepBoostedJetTags_probTbqq',
    'orig_pfMassDecorrelatedDeepBoostedJetTags_probTbc',
    'orig_pfMassDecorrelatedDeepBoostedJetTags_probTbq',
    'orig_pfMassDecorrelatedDeepBoostedJetTags_probWcq',
    'orig_pfMassDecorrelatedDeepBoostedJetTags_probWqq',
    'orig_pfMassDecorrelatedDeepBoostedJetTags_probZbb',
    'orig_pfMassDecorrelatedDeepBoostedJetTags_probZcc',
    'orig_pfMassDecorrelatedDeepBoostedJetTags_probZqq',
    'orig_pfMassDecorrelatedDeepBoostedJetTags_probHbb',
    'orig_pfMassDecorrelatedDeepBoostedJetTags_probHcc',
    'orig_pfMassDecorrelatedDeepBoostedJetTags_probHqqqq',
    'orig_pfMassDecorrelatedDeepBoostedJetTags_probQCDbb',
    'orig_pfMassDecorrelatedDeepBoostedJetTags_probQCDcc',
    'orig_pfMassDecorrelatedDeepBoostedJetTags_probQCDb',
    'orig_pfMassDecorrelatedDeepBoostedJetTags_probQCDc',
    'orig_pfMassDecorrelatedDeepBoostedJetTags_probQCDothers',
    'orig_pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags_TvsQCD',
    'orig_pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags_WvsQCD',
    'orig_pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags_ZHbbvsQCD',
    'orig_pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags_ZHccvsQCD',
    'orig_pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags_bbvsLight',
    'orig_pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags_ccvsLight',

    'orig_pfDeepDoubleBvLJetTags_probHbb',
    'orig_pfDeepDoubleCvBJetTags_probHcc',
    'orig_pfDeepDoubleCvLJetTags_probHcc',
    'orig_pfMassIndependentDeepDoubleBvLJetTags_probHbb',
    'orig_pfMassIndependentDeepDoubleCvBJetTags_probHcc',
    'orig_pfMassIndependentDeepDoubleCvLJetTags_probHcc',
    ]

def load_data(args):

    train_val_filelist = glob.glob(args.data_train)
    n_train = int(args.train_val_split * len(train_val_filelist))

    wgtvar = args.weight_names
    if wgtvar == '': wgtvar = None

    d = DataFormat(train_groups, train_vars, label_var, wgtvar, obs_vars, extra_label_vars=extra_label_vars, filename=train_val_filelist[0])

    logging.info('Using the following variables:\n' +
                 '\n'.join([v_group + '\n\t' + str(train_vars[v_group]) for v_group in train_groups ]))
    logging.info('Using weight\n' + str(wgtvar))

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
    out['scale_method'] = orig.get('scale_method', 'upper')
    out['input_names'] = groups
    out['input_shapes'] = shapes
    out['var_names'] = var_names
#     print(out)
    with open(output, 'w') as f:
        json.dump(out, f, indent=2, sort_keys=True)
    logging.info('Output json file to %s' % output)
