'''
Description on how to produce metadata file.
'''

treename = 'deepntuplizer/tree'
reweight_events = -1
reweight_bins = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650,
               700, 800, 900, 1000, 1100,
               1200, 1400, 1600, 5000]
metadata_events = 1000000
selection = 'jet_tightId && (fj_isTop && fj_pt>400) || (fj_isH && fj_pt>300) || (!fj_isTop && !fj_isH)'
var_groups = {
    # 'group_name': ( ('regex1', 'regex2', ...), list_length )
    'fjvars': (('fj_',), None),
    }
var_blacklist = [
    'fj_gen_pt',
    'fj_gen_eta',
    'fj_phi',
    'fj_mass',
    'fj_tau1',
    'fj_tau2',
    'fj_tau3',

    'fj_isBB',
    'fj_isNonBB',
    'fj_label',
    'fj_labelJMAR',

    'n_pfcands',
    'pfcand_VTX_ass',
    'pfcand_fromPV',
    'pfcand_lostInnerHits',
    'pfcand_dz',
    'pfcand_dzsig',
    'pfcand_dxy',
    'pfcand_dxysig',

    'n_tracks',

    'trackBTag_Momentum',
    'trackBTag_Eta',
    'trackBTag_PPar',

    'n_sv',
    ]
var_no_transform_branches = [
    'npv',
    'npfcands', 'ntracks', 'nsv',
    'fj_pt', 'fj_eta', 'fj_tau21', 'fj_tau32',
    'fj_sdmass', 'fj_n_sdsubjets', 'fj_doubleb',
    ]
label_list = ['fj_isLight', 'fj_isTop', 'fj_isW', 'fj_isZ', 'fj_isH']
reweight_var = 'fj_pt'
var_img = None
var_pos = None
n_pixels = None
img_ranges = None
