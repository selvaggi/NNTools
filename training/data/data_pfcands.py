from __future__ import print_function

from common.data import DataFormat, DataLoader
import glob
import logging

label_var = 'label'
train_groups = ['pfcand', 'track', 'sv']
train_vars = {}
train_vars['pfcand'] = (
    'pfcand_ptrel',
    'pfcand_erel',
    'pfcand_phirel',
    'pfcand_etarel',
    'pfcand_deltaR',
    'pfcand_puppiw',

    'pfcand_drminsv',
    'pfcand_drsubjet1',
    'pfcand_drsubjet2',

#     'pfcand_charge',
#     'pfcand_isMu',
#     'pfcand_isEl',
#     'pfcand_isGamma',
#     'pfcand_isChargedHad',
#     'pfcand_isNeutralHad',

    'pfcand_hcalFrac',

#     'pfcand_VTX_ass',
#     'pfcand_fromPV',
#     'pfcand_lostInnerHits',

#     'pfcand_dz',
#     'pfcand_dzsig',
#     'pfcand_dxy',
#     'pfcand_dxysig',
    )

train_vars['track'] = (
    'track_ptrel',
    'track_erel',
    'track_phirel',
    'track_etarel',
    'track_deltaR',
#     'track_puppiw',
#     'track_pt',

    'track_drminsv',
    'track_drsubjet1',
    'track_drsubjet2',

#     'track_charge',
#     'track_isMu',
#     'track_isEl',
#     'track_isChargedHad',

#     'track_VTX_ass',
#     'track_fromPV',
#     'track_lostInnerHits',

    'track_dz',
    'track_dzsig',
    'track_dxy',
    'track_dxysig',

    'track_normchi2',
    'track_quality',

    'track_dptdpt',
    'track_detadeta',
    'track_dphidphi',
    'track_dxydxy',
    'track_dzdz',
    'track_dxydz',
    'track_dphidxy',
    'track_dlambdadz',

    'trackBTag_EtaRel',
    'trackBTag_PtRatio',
    'trackBTag_PParRatio',
    'trackBTag_Sip2dVal',
    'trackBTag_Sip2dSig',
    'trackBTag_Sip3dVal',
    'trackBTag_Sip3dSig',
    'trackBTag_JetDistVal',
    )

train_vars['sv'] = (
    'sv_ptrel',
    'sv_erel',
    'sv_phirel',
    'sv_etarel',
    'sv_deltaR',
    'sv_pt',
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
    'orig_fj_pt',
    'orig_fj_eta',
    'orig_fj_sdmass',
    'orig_fj_n_sdsubjets',
    "orig_fj_doubleb",
    'orig_fj_tau21',
    'orig_fj_tau32',
    'orig_npv',
    'orig_npfcands',
    'orig_ntracks',
    'orig_nsv',
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

    if args.predict:
        test_filelist = glob.glob(args.data_test)
        test = DataLoader(test_filelist, d, batch_size=args.batch_size, predict_mode=True, shuffle=False, args=args)
        return test
    else:
        train = DataLoader(train_val_filelist[:n_train], d, batch_size=args.batch_size, args=args)
        val = DataLoader(train_val_filelist[n_train:], d, batch_size=args.batch_size, args=args)
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
