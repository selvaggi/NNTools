import os
import json
import FWCore.ParameterSet.Config as cms
import argparse

# https://stackoverflow.com/questions/956867/how-to-get-string-objects-instead-of-unicode-from-json
def json_load_byteified(file_handle):
    return _byteify(
        json.load(file_handle, object_hook=_byteify),
        ignore_dicts=True
    )

def json_loads_byteified(json_text):
    return _byteify(
        json.loads(json_text, object_hook=_byteify),
        ignore_dicts=True
    )

def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data


parser = argparse.ArgumentParser('Convert json to CMS PSet')
parser.add_argument('-i', '--input', 
    help='Input file.'
)
parser.add_argument('-o', '--output',
    help='Output file.'
)
args = parser.parse_args()

with open(args.input) as fp:
    j = json_load_byteified(fp)

cfg = cms.PSet()
cfg.input_names = cms.vstring(*j['input_names'])

for name in j['input_names']:
    p = cms.PSet(
        input_shape=cms.vuint32(j['input_shapes'][name]),
        var_names=cms.vstring(*j['var_names'][name]),
        var_length=cms.uint32(j['var_info'][j['var_names'][name][0]]['size']),
        )
    setattr(cfg, name, p)
    p.var_infos=cms.PSet()
    
    for v in j['var_names'][name]:
        info = j['var_info'][v]
        pvar = cms.PSet(
            median=cms.double(info['median']),
            upper=cms.double(info['upper']),
            )
        setattr(p.var_infos, v, pvar)

with open(args.output, 'w') as fout:
    fout.write('import FWCore.ParameterSet.Config as cms\n\n')
    fout.write('pfDeepBoostedJetPreprocessParams = '+str(cfg))
