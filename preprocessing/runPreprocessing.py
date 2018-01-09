'''
Script for data preprocessing.

@author: hqu
'''

from __future__ import print_function

import os
import math
import argparse

from metadata import Metadata
from converter import writeData
import multiprocessing
import functools

import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')


def create_metadata(args):
    fullpath = os.path.join(args.outputdir, args.metadata)
    if os.path.exists(fullpath):
        return

    from importlib import import_module
    d = import_module('data_formats.' + args.data_format)
    md = Metadata(inputdir=args.inputdir,
                  input_filter=d.input_filter,
                  treename=d.treename,
                  reweight_events=d.reweight_events,
                  reweight_bins=d.reweight_bins,
                  metadata_events=d.metadata_events,
                  selection=d.selection,
                  var_groups=d.var_groups,
                  var_blacklist=d.var_blacklist,
                  var_no_transform_branches=d.var_no_transform_branches,
                  label_list=d.label_list,
                  reweight_var=d.reweight_var,
                  reweight_classes=d.reweight_classes,
                  reweight_method=d.reweight_method,
                  var_img=d.var_img,
                  var_pos=d.var_pos,
                  n_pixels=d.n_pixels,
                  img_ranges=d.img_ranges,
                  )
    md.produceMetadata(fullpath)

def update_metadata(args):
    create_metadata(args)
    from importlib import import_module
    d = import_module('data_formats.' + args.data_format)
    md = Metadata(args.inputdir,
                  treename=d.treename,
                  reweight_events=d.reweight_events,
                  reweight_bins=d.reweight_bins,
                  metadata_events=d.metadata_events,
                  selection=d.selection,
                  var_groups=d.var_groups,
                  var_blacklist=d.var_blacklist,
                  var_no_transform_branches=d.var_no_transform_branches,
                  label_list=d.label_list,
                  reweight_var=d.reweight_var,
                  reweight_classes=d.reweight_classes,
                  var_img=d.var_img,
                  var_pos=d.var_pos,
                  n_pixels=d.n_pixels,
                  img_ranges=d.img_ranges,
                  )
    md.loadMetadata(os.path.join(args.outputdir, args.metadata))
    if args.remake_filelist:
        md.updateFilelist(args.test_sample)
    if args.remake_weights:
        md.updateWeights(args.test_sample)
    md.writeMetadata(os.path.join(args.jobdir, args.metadata))
    njobs = int(math.ceil(float(sum(md.num_events)) / args.events_per_file))
    return md,njobs

def submit(args):
    
    scriptfile = os.path.join(args.jobdir, 'runjob.sh')
    metadatafile = os.path.join(args.jobdir, args.metadata)

    if not args.resubmit:
        from helper import xrd
        md, njobs = update_metadata(args)

        script = \
'''#!/bin/bash
jobid=$1
workdir=`pwd`

echo `hostname`
echo "workdir: $workdir"
echo "args: $@"
ls -l

export PATH={conda_path}:$PATH
source activate {conda_env_name}
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

python {script} {outputdir} $jobid -n {events} {test_sample}
status=$?
echo "Status = $status"
ls -l

if [ $status -ne 0 ]; then
    exit $status
else
    echo
    {xrdcp}
fi

exit $status
'''.format(conda_path=args.conda_path,
           conda_env_name=args.conda_env_name,
           script=os.path.abspath('converter.py'),
           outputdir=args.outputdir,
           events=args.events_per_file,
           test_sample='--test-sample' if args.test_sample else '',
           xrdcp='' if not args.outputdir.startswith('/eos') else 'xrdcp -np *.h5 %s ; rm *.h5' % (xrd(args.outputdir) + '/')
           )

        with open(scriptfile, 'w') as f:
            f.write(script)
        os.system('chmod +x %s' % scriptfile)

        jobids = [str(jobid) for jobid in range(njobs)]
        jobids_file = os.path.join(args.jobdir, 'submit.txt')

    else:
        # resubmit
        jobids = []
        jobids_file = os.path.join(args.jobdir, 'resubmit.txt')
        log_files = [f for f in os.listdir(args.jobdir) if f.endswith('.log')]
        for fn in log_files:
            with open(os.path.join(args.jobdir, fn)) as logfile:
                errormsg = None
                for line in reversed(logfile.readlines()):
                    if 'Job removed' in line or 'aborted' in line:
                        errormsg = line
                    if 'Job submitted from host' in line:
                        # if seeing this first: the job has been resubmited
                        break
                    if 'return value' in line:
                        if 'return value 0' not in line:
                            errormsg = line
                        break
                if errormsg:
                    logging.debug(fn + '\n   ' + errormsg)
                    jobids.append(fn.split('.')[0])
                    assert jobids[-1].isdigit()

    with open(jobids_file, 'w') as f:
        f.write('\n'.join(jobids))

    condordesc = '''\
universe              = vanilla
requirements          = (Arch == "X86_64") && (OpSys == "LINUX")
request_disk          = 10000000
executable            = {scriptfile}
arguments             = $(jobid)
transfer_input_files  = {metadatafile}
output                = {jobdir}/$(jobid).out
error                 = {jobdir}/$(jobid).err
log                   = {jobdir}/$(jobid).log
use_x509userproxy     = true
+MaxRuntime           = 172800
Should_Transfer_Files = YES
queue jobid from {jobids_file}
'''.format(scriptfile=os.path.abspath(scriptfile),
           metadatafile=os.path.abspath(metadatafile),
           jobdir=os.path.abspath(args.jobdir),
           outputdir=args.outputdir,
           jobids_file=os.path.abspath(jobids_file)
    )
    condorfile = os.path.join(args.jobdir, 'submit.cmd')
    with open(condorfile, 'w') as f:
        f.write(condordesc)

    print('Run the following command to submit the jobs:\ncondor_submit {condorfile}'.format(condorfile=condorfile))

def run_all(args):
    md, njobs = update_metadata(args)
#     for jobid in range(njobs):
#         writeData(md, args.outputdir, jobid, batch_mode=False,
#                         test_sample=args.test_sample, events=args.events_per_file, dryrun=args.dryrun)
    pool = multiprocessing.Pool(args.nproc)
    pool.map(
        functools.partial(writeData, md, args.outputdir, batch_mode=False,
                          test_sample=args.test_sample, events=args.events_per_file, dryrun=args.dryrun),
        range(njobs)
        )

def main():
    parser = argparse.ArgumentParser('Preprocess ntuples')
    parser.add_argument('inputdir',
        help='Input diretory.'
    )
    parser.add_argument('outputdir',
        help='Output directory'
    )
    parser.add_argument('-d', '--data-format',
        default='ak8_pfcands_list',
        help='data format file. Default: %(default)s'
    )
    parser.add_argument('-m', '--metadata',
        default='metadata.json',
        help='Metadata json file. Default: %(default)s'
    )
    parser.add_argument('-t', '--submittype',
        default='condor', choices=['interactive', 'condor'],
        help='Method of job submission. [Default: %(default)s]'
        )
    parser.add_argument('--resubmit',
        action='store_true', default=False,
        help='Resubmit failed jobs. Default: %(default)s'
    )
    parser.add_argument('-j', '--jobdir',
        default='jobs',
        help='Directory for job files. [Default: %(default)s]'
        )
    parser.add_argument('--nproc',
        type=int, default=8,
        help='Number of jobs to run in parallel. Default: %(default)s'
    )
    parser.add_argument('-n', '--events-per-file',
        type=int, default=50000,
        help='Number of input files to process in one job. Default: %(default)s'
    )
    parser.add_argument('--test-sample',
        action='store_true', default=False,
        help='Convert test data. Default: %(default)s'
    )
    parser.add_argument('--remake-filelist',
        action='store_true', default=False,
        help='Remake filelist. Default: %(default)s'
    )
    parser.add_argument('--remake-weights',
        action='store_true', default=False,
        help='Remake reweighting weights. Default: %(default)s'
    )
    parser.add_argument('--conda-path',
        default='~/miniconda2/bin',
        help='Conda bin path. Default: %(default)s'
    )
    parser.add_argument('--conda-env-name',
        default='prep',
        help='Conda env name. Default: %(default)s'
    )
    parser.add_argument('--dryrun',
        action='store_true', default=False,
        help='Do not convert -- only produce metadata. Default: %(default)s'
    )
#     parser.add_argument('--debug',
#         action='store_true', default=False,
#         help='Run in debug mode. Default: %(default)s'
#     )
    args = parser.parse_args()
#     print(args)

    if not os.path.exists(args.jobdir):
        os.makedirs(args.jobdir)
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    if args.submittype == 'interactive':
        run_all(args)
    elif args.submittype == 'condor':
        submit(args)

if __name__ == '__main__':
    main()
