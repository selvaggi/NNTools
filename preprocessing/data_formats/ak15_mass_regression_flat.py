'''
Description on how to produce metadata file.
'''

input_filter = None
treename = 'deepntuplizer/tree'
selection = '''jet_tightId && jet_no<2 && fj_gen_mass>20 && fj_pt>170 && fj_pt<1200'''

reweight_events = -1
reweight_bins = [
    [170, 203, 242, 289, 346, 413, 493, 589, 704, 841, 1004, 1200],  # pt: np.exp(np.linspace(np.log(170), np.log(1200), 12)).astype(np.int32)
    list(range(20, 251, 10))  # mass
    ]
reweight_var = ['fj_pt', 'fj_gen_mass']
reweight_classes = [
    'label_H_bb', 'label_H_cc', 'label_H_qq',
    ]
reweight_method = 'flat'

metadata_events = 1000000
var_groups = {
    # 'group_name': ( ('regex1', 'regex2', ...), list_length )
    'pfcand': (('pfcand_',), 100),
    'sv': (('sv_',), 7),
    }
var_blacklist = [
    'fj_gen_pt',
    'fj_gen_eta',

    'fj_isBB',
    'fj_isNonBB',

    'n_pfcands',
    'n_tracks',
    'n_sv',

    'pfcand_btagJetDistSig',
    ]
var_no_transform_branches = [
    'event_no', 'jet_no',
    'fj_label',
    'fj_isQCD', 'fj_isTop', 'fj_isW', 'fj_isZ', 'fj_isH',
    'npv',
    'n_pfcands', 'n_sv',
    'fj_pt', 'fj_eta', 'fj_phi', 'fj_mass',
    'fj_n_sdsubjets', 'fj_nbHadrons', 'fj_ncHadrons',
    'fj_genjet_pt', 'fj_genjet_mass', 'fj_genjet_sdmass',
    'fj_gen_mass',

    'pfMassDecorrelatedDeepBoostedJetTags_probTbcq',
    'pfMassDecorrelatedDeepBoostedJetTags_probTbqq',
    'pfMassDecorrelatedDeepBoostedJetTags_probTbc',
    'pfMassDecorrelatedDeepBoostedJetTags_probTbq',
    'pfMassDecorrelatedDeepBoostedJetTags_probWcq',
    'pfMassDecorrelatedDeepBoostedJetTags_probWqq',
    'pfMassDecorrelatedDeepBoostedJetTags_probZbb',
    'pfMassDecorrelatedDeepBoostedJetTags_probZcc',
    'pfMassDecorrelatedDeepBoostedJetTags_probZqq',
    'pfMassDecorrelatedDeepBoostedJetTags_probHbb',
    'pfMassDecorrelatedDeepBoostedJetTags_probHcc',
    'pfMassDecorrelatedDeepBoostedJetTags_probHqqqq',
    'pfMassDecorrelatedDeepBoostedJetTags_probQCDbb',
    'pfMassDecorrelatedDeepBoostedJetTags_probQCDcc',
    'pfMassDecorrelatedDeepBoostedJetTags_probQCDb',
    'pfMassDecorrelatedDeepBoostedJetTags_probQCDc',
    'pfMassDecorrelatedDeepBoostedJetTags_probQCDothers',
    'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags_TvsQCD',
    'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags_WvsQCD',
    'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags_ZHbbvsQCD',
    'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags_ZHccvsQCD',

    "fj_tau21",
    "fj_tau32",
    "fj_sdmass",
    "fj_sdmass_fromsubjets",
    "fj_rho",
    "fj_corrsdmass",
    ]
label_list = ['label_H_bb', 'label_H_cc', 'label_H_qq',
              'label_QCD_bb', 'label_QCD_cc', 'label_QCD_b', 'label_QCD_c', 'label_QCD_others',
              ]

scale_method = 'max'
var_img = None
var_pos = None
n_pixels = None
img_ranges = None
