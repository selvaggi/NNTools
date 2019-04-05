'''
Class to produce metadata for root to numpy conversion.

@author: hqu
'''

from __future__ import print_function

import os
import re
import json
import logging
import numpy as np
import pandas as pd

from helper import get_num_events

class Metadata(object):

    ''' Compile the metadata. '''

    def __init__(self,
                 inputdir,
                 input_filter=None,
                 treename='deepntuplizer/tree',
                 reweight_events=100000,
                 reweight_bins=[[200, 5000], [-999, 999]],
                 metadata_events=100000,
                 selection=None,
                 var_groups=None,  # {group_name:(regex, num)}
                 var_blacklist=None,
                 var_no_transform_branches=None,
                 label_list=None,
                 reweight_var=['fj_pt', 'fj_sdmass'],
                 reweight_classes=['fj_isTop', 'fj_isW', 'fj_isZ', 'fj_isH', 'fj_isQCD'],
                 reweight_method='flat',
                 scale_method='upper',
                 var_img='pfcand_ptrel',
                 var_pos=['pfcand_etarel', 'pfcand_phirel'],
                 n_pixels=64,
                 img_ranges=[[-0.8, 0.8], [-0.8, 0.8]],
                 ):
        self._inputdir = inputdir  # data members starting with '_' is not loaded from json

        self.input_filter = input_filter  # regex for filtering the input paths
        self.treename = treename
        self.reweight_var = reweight_var
        self.reweight_classes = reweight_classes
        self._reweight_events = reweight_events
        self._reweight_bins = reweight_bins
        self._metadata_events = metadata_events
        self.selection = selection
        self.var_groups = var_groups
        self.var_blacklist = var_blacklist
        self.var_no_transform_branches = var_no_transform_branches
        self.label_branches = label_list
        
        self.reweight_method = reweight_method.lower()
        if self.reweight_method not in ['none', 'flat', 'ref']:
            raise NotImplemented('reweight method %s not recognized' % reweight_method)

        self.scale_method = scale_method.lower()
        if self.scale_method not in ['upper', 'lower', 'average', 'max']:
            raise NotImplemented('scale method %s not recognized' % scale_method)

        self.var_img = var_img
        self.var_pos = var_pos
        self.n_pixels = n_pixels
        self.img_ranges = img_ranges

        self.inputfiles = None
        self.num_events = None

    def produceMetadata(self, filepath):
        logging.info('Start producing metadata...')
        # make file list
        self.updateFilelist()
        # make var list
        self._make_varlist()
        # make weights
        self._make_weights()
        # make transfromation info
        self._make_infos()
        # write metadata
        self.writeMetadata(filepath)

    def loadMetadata(self, filepath, override=True):
        with open(filepath) as metafile:
            md = json.load(metafile, encoding='ascii')
            for k in md:
                if k.startswith('_'): continue
                if override:
                    setattr(self, k, md[k])
                else:
                    if not hasattr(self, k) or getattr(self, k) is None:
                        setattr(self, k, md[k])
        logging.info('Metadata loaded from ' + filepath)

    def updateFilelist(self, test_sample=False):
        import re
        self.inputfiles = []
        self.num_events = []
        counter = 0
        for dp, dn, filenames in os.walk(self._inputdir, followlinks=True):
            if 'failed' in dp or 'ignore' in dp:
                continue
            if self.input_filter and re.search(self.input_filter, dp):
                logging.debug('Ignoring inputdir %s', dp)
                continue
#             if not test_sample and 'test_sample' in dp:
#                 # train/val samples
#                 continue
#             if test_sample and 'test_sample' not in dp:
#                 # test samples
#                 continue
            for f in filenames:
                if not f.endswith('.root'):
                    continue
                fullpath = os.path.realpath(os.path.join(dp, f))
                nevts = get_num_events(fullpath, self.treename)
                if nevts:
                    self.inputfiles.append(fullpath)
                    self.num_events.append(nevts)
                    counter += 1
                    if counter%10==0:
                        logging.debug('%d files processed...' % counter)
                else:
                    logging.warning('Ignore erroneous file %s' % fullpath)
        self._total_events = sum(self.num_events)
        logging.info('Created file list from directory %s\nFiles:%d, Events:%d' % (self._inputdir, len(self.inputfiles), self._total_events))
        return (self.inputfiles, self.num_events)

    def updateWeights(self, test_sample=False):
        if test_sample:
            return
        else:
            self._make_weights()

    def writeMetadata(self, filepath):
        with open(filepath, 'w') as metafile:
            json.dump(self.__dict__, metafile, indent=2, encoding='ascii', sort_keys=True)
        logging.info('Metadata written to ' + filepath)


    def _make_varlist(self):
        # get all branches and filter them using input variable list
        from root_numpy import root2array
        df = pd.DataFrame(root2array(self.inputfiles[0], treename=self.treename, stop=1))
        self._all_branches = df.columns.values.tolist()
        self.var_branches = []
        self.var_sizes = {}
        for k in self._all_branches:
            matched = False
            for v_group in self.var_groups:
                size = self.var_groups[v_group][1]
                for regex in self.var_groups[v_group][0]:
                    if re.match(regex, k):
                        self.var_branches.append(k)
                        self.var_sizes[k] = size
                        matched = True
                        break
                if matched: break
        for var in self.var_blacklist + self.label_branches + self.reweight_classes:
            try:
                self.var_branches.remove(var)
            except ValueError:
                pass
        logging.info('Training vars:\n' + '\n'.join(self.var_branches))
        # check no_transform vars
        _var_no_transform = []
        for v in self.var_no_transform_branches:
            if v in self._all_branches:
                _var_no_transform.append(v)
            else:
                logging.warning('No-transform var %s not found in the input. Will be ignored!' % v)
        self.var_no_transform_branches = _var_no_transform


    def _prepare_reweight_info(self, rec):
        ''' Produce metadata for reweighting. Goal:
            1) Produce flat pT spectrum.
            2) Balance the class weights on top of that
        '''
        class_events = {}
        result = {}
        for label in self.reweight_classes:
            pos = (rec[label] == 1)
            x = np.minimum(rec[self.reweight_var[0]][pos], max(self._reweight_bins[0]))
            y = np.minimum(rec[self.reweight_var[1]][pos], max(self._reweight_bins[1]))
#             class_events[label] = 0
            hist, x_edges, y_edges = np.histogram2d(x, y, bins=self._reweight_bins)
            hist = np.asfarray(hist, dtype=np.float32)
            result[label] = {'x_edges':x_edges.tolist(), 'y_edges':y_edges.tolist(), 'hist':hist, 'raw_hist':hist[:].tolist()}
            logging.debug('%s:\n%s' % (label, str(hist)))
#             if min(hist[-2:]) < 10:
#                 logging.warning('Not enough events for cateogry %s:\n%s' % (label, str(hist)))
#                 raise RuntimeError('Not enough events for cateogry %s:\n%s' % (label, str(hist)))
        if self.reweight_method == 'flat':
            for label in self.reweight_classes:
                hist = result[label]['hist']
                hist_non_zero = hist[hist > 0]
                min_val = np.min(hist_non_zero)
                med_val = np.median(hist)
                ref_val = np.percentile(hist_non_zero, 10)
                logging.debug('label:%s, median=%f, min=%f, ref=%f, ref/min=%f' % (label, med_val, min_val, ref_val, ref_val / min_val))
                class_events[label] = ref_val
                wgt = ref_val / hist  # will produce inf if hist[ix,iy]=0
                wgt[np.isinf(wgt)] = 0  # get rid of inf
                wgt = np.clip(wgt, 0, 5)
                result[label]['hist'] = wgt.tolist()
            min_nevt = min(class_events.values())
            for label in self.reweight_classes:
                class_wgt = float(min_nevt) / class_events[label]
                result[label]['class_wgt'] = class_wgt
        elif self.reweight_method == 'ref':
            # use class 0 as the reference
            # will get both shape wgt and class wgt at the same time
            hist_ref = result[self.reweight_classes[0]]['hist']
            upper_wgt = 1.0
            for label in self.reweight_classes:
                wgt = hist_ref / result[label]['hist']
                wgt[np.isinf(wgt)] = 0  # get rid of inf
                upper = np.percentile(wgt, 90)
                if upper > upper_wgt:
                    upper_wgt = upper
                result[label]['hist'] = wgt
            # rescale the weights to make them less than 1
            for label in self.reweight_classes:
                result[label]['hist'] = (result[label]['hist'] / upper_wgt).tolist()
                result[label]['class_wgt'] = 1
        return result

    def _make_weights(self):
        if self.reweight_method == 'none':
            logging.info('-- Reweighting is disabled --')
            return
        logging.info('Start making weights...\n Var: %s\n Classes: %s\n Selection: %s' % (str(self.reweight_var), str(self.reweight_classes), self.selection))
        # fraction of events to take from each file
        from root_numpy import root2array
        frac = 1.0
        if self._reweight_events > 0:
            frac = float(self._reweight_events) / self._total_events
        if frac < 1:
            pieces = []
            for fn, n in zip(self.inputfiles, self.num_events):
                a = root2array(fn, treename=self.treename, selection=self.selection, stop=int(frac * n),
                               branches=self.reweight_classes + self.reweight_var)
                pieces.append(a)
            rec = np.concatenate(pieces)
        else:

            rec = root2array(self.inputfiles, treename=self.treename, selection=self.selection,
                               branches=self.reweight_classes + self.reweight_var)
        logging.info('Use %d events to produce reweight info, selection:\n%s' % (rec.shape[0], self.selection))
        # get distribution for reweighting
        self.reweight_info = self._prepare_reweight_info(rec)
        logging.debug('Reweight info:\n' + str(self.reweight_info))

    def _make_infos(self):
        # make variables transformation infos
        from root_numpy import root2array
        frac = 1.0
        _inputfiles = self.inputfiles
        _num_events = self.num_events
        if self._metadata_events > 0:
            nfiles = int(5 * float(self._metadata_events) / self._total_events * len(self.inputfiles))
            file_inds = np.arange(len(self.inputfiles))
            np.random.shuffle(file_inds)
            file_inds = file_inds[:nfiles]
            _inputfiles = [self.inputfiles[i] for i in file_inds]
            _num_events = [self.num_events[i] for i in file_inds]
            frac = float(self._metadata_events) / sum(_num_events)

        first = True

        self.branches_info = {}
        for var in self.var_branches:
            var_size = self.var_sizes[var]
            pieces = []
            for fn, n in zip(_inputfiles, _num_events):
                v = root2array(fn, treename=self.treename, selection=self.selection,
                               branches=var, stop=int(frac * n))
                pieces.append(v)
            a = np.concatenate(pieces)
            if first:
                first = False
                logging.debug('Use %d events from %d files for var transform info' % (a.shape[0], len(_inputfiles)))
            size = None
            if a.dtype == np.object:
                if var_size:
                    size = var_size  # use given size if provided
                else:
                    lengths = [len(row) for row in a]
                    size = int(round(np.percentile(lengths, 95)))  # else get 95% percentile of the length
                a = np.nan_to_num(np.concatenate(a))  # then flatten vector vars for calculations
            else:
                a = np.nan_to_num(a)
            self.branches_info[var] = {
                'size'  : size,
                'median': float(np.percentile(a, 50)),  # need float otherwise cannot serialize to json
                'lower' : float(np.percentile(a, 16)),
                'upper' : float(np.percentile(a, 84)),
                'min'   : float(np.min(a)),
                'max'   : float(np.max(a)),
                'mean'  : float(np.mean(a)),
                'std'   : float(np.std(a)),
                }
            logging.debug(var + ': ' + str(self.branches_info[var]))
