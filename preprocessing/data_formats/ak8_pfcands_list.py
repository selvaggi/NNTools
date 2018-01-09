'''
Description on how to produce metadata file.
'''

input_filter = None
treename = 'deepntuplizer/tree'
reweight_events = -1
reweight_bins = [list(range(200, 2051, 50)), [-10000, 10000]]
# reweight_bins = [list(range(200, 2051, 50)), list(range(0, 251, 10))] # reweight in (pt, mass)
metadata_events = 1000000
selection = '''jet_tightId \
&& ( !label_H_cc )'''
# && ( (sample_isQCD && fj_isQCD) || (!sample_isQCD && !fj_isQCD)) \
var_groups = {
    # 'group_name': ( ('regex1', 'regex2', ...), list_length )
    'pfcand': (('pfcand_',), 100),
    'track': (('track_', 'trackBTag_',), 60),
    'sv': (('sv_',), 5),
    }
var_blacklist = [
    'fj_gen_pt',
    'fj_gen_eta',
    'fj_tau1',
    'fj_tau2',
    'fj_tau3',

    'fj_isBB',
    'fj_isNonBB',

    'n_pfcands',
    'pfcand_VTX_ass',
    'pfcand_fromPV',
    'pfcand_lostInnerHits',
    'pfcand_dz',
    'pfcand_dzsig',
    'pfcand_dxy',
    'pfcand_dxysig',

    'n_tracks',

#     'trackBTag_Momentum',
#     'trackBTag_Eta',
#     'trackBTag_PPar',

    'n_sv',
    ]
var_no_transform_branches = [
    'fj_labelJMAR', 'fjJMAR_gen_pt', 'fjJMAR_gen_eta', 'fjJMAR_gen_pdgid',
    'fj_label',
    'fj_isQCD', 'fj_isTop', 'fj_isW', 'fj_isZ', 'fj_isH',
    'npv',
    'n_pfcands', 'n_tracks', 'n_sv',
    'fj_pt', 'fj_eta', 'fj_phi', 'fj_mass',
    'fj_tau21', 'fj_tau32',
    'fj_sdmass', 'fj_n_sdsubjets',
    'fjPuppi_tau21', 'fjPuppi_tau32', 'fjPuppi_corrsdmass',
    'fj_doubleb', 'pfCombinedInclusiveSecondaryVertexV2BJetTags', 'fj_sdsj1_csv', 'fj_sdsj2_csv'
    ]
# label_list = ['fj_isQCD', 'fj_isTop', 'fj_isW', 'fj_isZ', 'fj_isH']
label_list = ['label_Top_bcq', 'label_Top_bqq', 'label_Top_bc', 'label_Top_bq',
              'label_W_cq', 'label_W_qq',
              'label_Z_bb', 'label_Z_cc', 'label_Z_qq',
              'label_H_bb', 'label_H_qqqq',
              'label_QCD_bb', 'label_QCD_cc', 'label_QCD_b', 'label_QCD_c', 'label_QCD_others',
              ]
reweight_var = ['fj_pt', 'fj_sdmass']
reweight_classes = ['fj_isTop', 'fj_isW', 'fj_isZ', 'fj_isH', 'fj_isQCD']
reweight_method = 'flat'
var_img = None
var_pos = None
n_pixels = None
img_ranges = None
