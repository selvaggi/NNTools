'''
Description on how to produce metadata file.
'''

input_filter = r'Tau|htata'
treename = 'deepntuplizer/tree'
reweight_events = -1
reweight_bins = [
    [200, 251, 316, 398, 501, 630, 793, 997, 1255, 1579, 1987, 2499],  # pt: np.exp(np.linspace(np.log(200), np.log(2500), 12)).astype(np.int32)
    list(range(30, 251, 10))  # mass
    ]
metadata_events = 1000000
selection = '''jet_tightId && fj_sdmass_fromsubjets>30'''
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
    'event_no',
    'fj_labelJMAR', 'fjJMAR_gen_pt', 'fjJMAR_gen_eta', 'fjJMAR_gen_pdgid',
    'fj_label',
    'fj_isQCD', 'fj_isTop', 'fj_isW', 'fj_isZ', 'fj_isH',
    'npv',
    'n_pfcands', 'n_sv',
    'fj_pt', 'fj_eta', 'fj_phi', 'fj_mass',
    'fj_n_sdsubjets', 'fj_nbHadrons', 'fj_ncHadrons',
    'fj_doubleb',

    'pfDeepBoostedJetTags_probTbcq',
    'pfDeepBoostedJetTags_probTbqq',
    'pfDeepBoostedJetTags_probTbc',
    'pfDeepBoostedJetTags_probTbq',
    'pfDeepBoostedJetTags_probWcq',
    'pfDeepBoostedJetTags_probWqq',
    'pfDeepBoostedJetTags_probZbb',
    'pfDeepBoostedJetTags_probZcc',
    'pfDeepBoostedJetTags_probZqq',
    'pfDeepBoostedJetTags_probHbb',
    'pfDeepBoostedJetTags_probHcc',
    'pfDeepBoostedJetTags_probHqqqq',
    'pfDeepBoostedJetTags_probQCDbb',
    'pfDeepBoostedJetTags_probQCDcc',
    'pfDeepBoostedJetTags_probQCDb',
    'pfDeepBoostedJetTags_probQCDc',
    'pfDeepBoostedJetTags_probQCDothers',
    'pfDeepBoostedDiscriminatorsJetTags_TvsQCD',
    'pfDeepBoostedDiscriminatorsJetTags_WvsQCD',
    'pfDeepBoostedDiscriminatorsJetTags_ZvsQCD',
    'pfDeepBoostedDiscriminatorsJetTags_ZbbvsQCD',
    'pfDeepBoostedDiscriminatorsJetTags_HbbvsQCD',
    'pfDeepBoostedDiscriminatorsJetTags_H4qvsQCD',
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
    'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags_bbvsLight',
    'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags_ccvsLight',

    'pfDeepDoubleBvLJetTags_probHbb',
    'pfDeepDoubleCvBJetTags_probHcc',
    'pfDeepDoubleCvLJetTags_probHcc',
    'pfMassIndependentDeepDoubleBvLJetTags_probHbb',
    'pfMassIndependentDeepDoubleCvBJetTags_probHcc',
    'pfMassIndependentDeepDoubleCvLJetTags_probHcc',

    "fj_tau21",
    "fj_tau32",
    "fj_sdmass",
    "fj_sdmass_fromsubjets",
    "fj_corrsdmass",

    'fj_z_ratio',
    'fj_trackSipdSig_3',
    'fj_trackSipdSig_2',
    'fj_trackSipdSig_1',
    'fj_trackSipdSig_0',
    'fj_trackSipdSig_1_0',
    'fj_trackSipdSig_0_0',
    'fj_trackSipdSig_1_1',
    'fj_trackSipdSig_0_1',
    'fj_trackSip2dSigAboveCharm_0',
    'fj_trackSip2dSigAboveBottom_0',
    'fj_trackSip2dSigAboveBottom_1',
    'fj_tau1_trackEtaRel_0',
    'fj_tau1_trackEtaRel_1',
    'fj_tau1_trackEtaRel_2',
    'fj_tau0_trackEtaRel_0',
    'fj_tau0_trackEtaRel_1',
    'fj_tau0_trackEtaRel_2',
    'fj_tau_vertexMass_0',
    'fj_tau_vertexEnergyRatio_0',
    'fj_tau_vertexDeltaR_0',
    'fj_tau_flightDistance2dSig_0',
    'fj_tau_vertexMass_1',
    'fj_tau_vertexEnergyRatio_1',
    'fj_tau_flightDistance2dSig_1',
    'fj_jetNTracks',
    'fj_nSV',
    ]
label_list = ['label_Top_bcq', 'label_Top_bqq', 'label_Top_bc', 'label_Top_bq',
              'label_W_cq', 'label_W_qq',
              'label_Z_bb', 'label_Z_cc', 'label_Z_qq',
              'label_H_bb', 'label_H_cc', 'label_H_qqqq',
              'label_QCD_bb', 'label_QCD_cc', 'label_QCD_b', 'label_QCD_c', 'label_QCD_others',
              ]
reweight_var = ['fj_pt', 'fj_sdmass_fromsubjets']
reweight_classes = [
    'fj_isTop',
    'fj_isW',
    'fj_isZ',
    'label_H_bb', 'label_H_cc', 'label_H_qqqq',  # H_bb, H_cc class_wgt divided by 2, H_qqqq devided by ~4-5 to become 1
    'fj_isQCD',
    ]
reweight_method = 'flat'
scale_method = 'max'
var_img = None
var_pos = None
n_pixels = None
img_ranges = None
