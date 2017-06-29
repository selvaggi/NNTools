'''
Convert root files to pyTables.

@author: hqu
'''

from __future__ import print_function

import os
import argparse
import numpy as np
import numexpr as ne
import math

import logging
from helper import xrd, pad_sequences

import tables
filters = tables.Filters(complevel=7, complib='blosc')
tables.set_blosc_max_threads(6)

def _write_carray(a, h5file, name, group_path='/', **kwargs):
    h5file.create_carray(group_path, name, obj=a, filters=filters, createparents=True, **kwargs)

def _make_labels(md, rec, h5file, name='label'):
    label = np.stack([rec[v] for v in md.label_branches], axis=1)
    _write_carray(label, h5file, name=name, title=','.join(md.label_branches))

def _make_weight(md, rec, h5file, name='weight'):
    wgt = np.zeros(rec.shape[0], dtype=np.float32)
    for label in md.label_branches:
        info = md.reweight_info[label]
        loc = rec[label] == 1
        rwgt_by = rec[md.reweight_var][loc]
        indices = np.clip(np.digitize(rwgt_by, info['bin_edges']) - 1, a_min=0, a_max=len(info['bin_edges']) - 2)
        wgt[loc] = np.asarray(info['hist'])[indices]
    _write_carray(wgt, h5file, name)

def _make_class_weight(md, rec, h5file, name='class_weight'):
    wgt = np.zeros(rec.shape[0], dtype=np.float32)
    for label in md.label_branches:
        loc = rec[label][:] == 1
        wgt[loc] = md.reweight_info[label]['class_wgt']
    _write_carray(wgt, h5file, name)

def _transform_var(md, rec, h5file, cols, no_transform=False, pad_method='zero'):
    for var in cols:
        var = str(var)  # get rid of unicode
        if no_transform:
            logging.debug('Writing variable orig_%s without transformation' % var)
            _write_carray(rec[var], h5file, name='orig_%s' % var)
            continue
        logging.debug('Transforming variable %s' % var)
        info = md.branches_info[var]
        median = np.float32(info['median'])
        scale = np.float32(info['upper'] - info['median'])
        if scale == 0:
            scale = 1
        if info['size'] and info['size'] > 1:
            # for sequence-like vars, perform padding first
            if pad_method == 'min':
                pad_value = info['min'] - scale  # ->min-1 ## FIXME: which is the better padding value
            elif pad_method == 'max':
                pad_value = info['max'] + scale  # ->max+1 ## FIXME: which is the better padding value
            elif pad_method == 'zero':
                pad_value = info['median']  # ->0 ## FIXME: which is the better padding value
            else:
                raise NotImplemented('pad_method %s is not supported' % pad_method)
            a = pad_sequences(rec[var], maxlen=info['size'], dtype='float32', padding='post', truncating='post', value=pad_value)
            a = np.nan_to_num(a)  # FIXME: protect against NaN
        else:
            a = rec[var].copy()  # need to copy, otherwise modifying the original array
        ne.evaluate('(a-median)/scale', out=a)
        _write_carray(a, h5file, name=var)

def _make_var(md, ct):
    pass

def _make_image(md, rec, h5file, output='img'):
    wgt = rec[md.var_img]
    x = rec[md.var_pos[0]]
    y = rec[md.var_pos[1]]
    img = np.zeros(shape=(len(wgt), md.n_pixels, md.n_pixels))
    for i in range(len(wgt)):
        hist2d, xedges, yedges = np.histogram2d(x[i], y[i], bins=[md.n_pixels, md.n_pixels], range=md.img_ranges, weights=wgt[i])
        img[i] = hist2d
    _write_carray(img, h5file, name=output)

def writeData(md, outputdir, jobid, batch_mode=False, test_sample=False, events=200000, dryrun=False):
    ''' Convert input files to a HDF file. '''
    from root_numpy import root2array

    def _write(rec, output):
        logging.debug(log_prefix + 'Start making output file')
        with tables.open_file(output, mode='w') as h5file:
            _make_labels(md, rec, h5file)
            logging.debug(log_prefix + 'Start producing weights')
            _make_weight(md, rec, h5file)
            _make_class_weight(md, rec, h5file)
            logging.debug(log_prefix + 'Start transforming variables')
            _transform_var(md, rec, h5file, md.var_no_transform_branches, no_transform=True)
            _transform_var(md, rec, h5file, md.var_branches)
            if md.var_img:
                logging.debug(log_prefix + 'Start making images')
                _make_image(md, rec, h5file, output='img')

    log_prefix = '[%d] ' % jobid
    outname = '{type}_file_{jobid}.h5'.format(type='test' if test_sample else 'train', jobid=jobid)
    output = os.path.join(outputdir, outname)
    if os.path.exists(output) and os.path.getsize(output) > 100 * 1024 * 1024:
        # ignore if > 100M
        logging.info(log_prefix + 'File %s already exist! Skipping.' % output)
        return

    frac = float(events) / sum(md.num_events)
    use_branches = set(md.var_branches + md.var_no_transform_branches + md.label_branches + [md.reweight_var])
    if md.var_img:
        use_branches += set([md.var_img] + md.var_pos)
#     use_branches = [str(var) for var in use_branches]
    logging.debug(log_prefix + 'Start loading from root files')

    pieces = []
    for fn, n in zip(md.inputfiles, md.num_events):
        step = int(math.ceil(frac * n))
        start = step * jobid
        stop = start + step
        if start >= n:
            continue
        filepath = xrd(fn) if batch_mode else fn
#         logging.debug('Load events [%d, %d) from file %s' % (start, stop, filepath))
        a = root2array(filepath, treename=md.treename, selection=md.selection, branches=use_branches, start=start, stop=stop)
        pieces.append(a)
    rec = np.concatenate(pieces)
    if rec.shape[0] == 0:
        return
    if not test_sample:
        # important: shuffle the array if not for testing
        np.random.shuffle(rec)

    if batch_mode:
        if not dryrun:
            _write(rec, outname)
        logging.info(log_prefix + 'Writing output to: \n' + outname)
    else:
        output_tmp = output + '.tmp'
        if not dryrun:
            _write(rec, output_tmp)
            os.rename(output_tmp, output)
        logging.info(log_prefix + 'Writing output to: \n' + output)

    logging.info(log_prefix + 'Done!')


def batch_write(args):
    from metadata import Metadata
    md = Metadata(None)
    md.loadMetadata(args.metadata)
    writeData(md, outputdir=args.outputdir, jobid=args.jobid, batch_mode=True,
              test_sample=args.test_sample, events=args.events_per_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert root files to PyTables.')
    parser.add_argument('-m', '--metadata',
        default='metadata.json',
        help='Path to the metadata file. Default:%(default)s')
    parser.add_argument('-n', '--events-per-file',
        type=int, default=50000,
        help='Number of events in each output file. Default: %(default)s'
        )
    parser.add_argument('--test-sample',
        action='store_true', default=False,
        help='Convert testing data instead of training/validation data. Default: %(default)s'
    )
    parser.add_argument('outputdir', help='Output directory for the metadata files.')
    parser.add_argument('jobid', type=int, help='Index of the output job.')

    args = parser.parse_args()
    batch_write(args)
