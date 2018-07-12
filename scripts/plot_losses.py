import os
import re
import argparse
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


fptn = r'[-+]?[0-9]*\.?[0-9]+'

train_epoch_batch_pattern = r'Epoch\[(\d+)\] Batch \[(\d+)\]'

loss_pattern = r'cross-entropy=(%s|nan)' % fptn
val_epoch_loss_pattern = r'Epoch\[(\d+)\] Validation-cross-entropy=(%s|nan)' % fptn
ylabel = 'Loss'

# loss_pattern = r'accuracy=(%s|nan)' % fptn
# val_epoch_loss_pattern = r'Epoch\[(\d+)\] Validation-accuracy=(%s|nan)' % fptn
# ylabel = 'Accuracy'

# loss_pattern = r'sec\s+mse=(%s|nan)' % fptn
# val_epoch_loss_pattern = r'Epoch\[(\d+)\] Validation-mse=(%s|nan)' % fptn
# ylabel = 'MSE'

# loss_pattern = r'[0-9]\s+mae=(%s|nan)' % fptn
# val_epoch_loss_pattern = r'Epoch\[(\d+)\] Validation-mae=(%s|nan)' % fptn
# ylabel = 'MAE'

def read_file(inputfile):
    train_loss = []
    val_loss = []
    batch_size = None
    is_val = False
    with open(inputfile) as f:
        for line in f:
            # first get batch size
            if batch_size is None and 'Epoch[0] Batch' in line:
                batch_size = int(re.search(r'Epoch\[0\] Batch \[(\d+)\]', line).groups()[0])
                print 'batch_size =', batch_size
            if 'Saved checkpoint to' in line or 'Saving model parameter' in line:
                is_val = True
                continue
            r = re.search(train_epoch_batch_pattern, line)
            if r and not is_val:
                epoch, batch = r.groups()
                epoch = int(epoch)
                batch = int(batch)
                if epoch >= len(train_loss):
                    # new epoch
                    train_loss.append([])
                batch_pos = batch / batch_size - 1
#                 print line
#                 print epoch, len(train_loss)
#                 print batch, len(train_loss[epoch])
                if batch_pos < len(train_loss[epoch]):
                    # in normal case, batch_pos should be equal to len(train_loss[epoch])
                    # if here, reset
                    assert batch_pos == 0
                    train_loss[epoch] = []
                loss = float(re.search(loss_pattern, line).groups()[0])
                train_loss[epoch].append(loss)
            else:
                v = re.search(val_epoch_loss_pattern, line)
                if v:
                    is_val = False
                    epoch, loss = v.groups()
                    epoch = int(epoch)
                    loss = float(loss)
                    if epoch < len(val_loss):
                        # if error happened
                        val_loss[epoch] = loss
                    else:
                        val_loss.append(loss)
#     print train_loss
#     print val_loss
    return train_loss, val_loss

def plot(train_loss, val_loss, output):
    train_x = []
    for epoch in range(len(train_loss)):
        x = np.arange(len(train_loss[epoch])) / float(len(train_loss[epoch]))
        x += epoch
        train_x.append(x)

    train_loss = np.concatenate(train_loss)
    plt.figure(figsize=(20, 5))
    plt.plot(np.concatenate(train_x), train_loss, '--', label='train')
    plt.plot(1 + np.arange(len(val_loss)), val_loss, label='val')

    ymin = 0.98 * np.min(train_loss)
    ymax = 1.1 * np.percentile(train_loss, q=80)

    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend(loc='best')
#     plt.yscale('log')
#     plt.ylim(ymin, ymax)
    plt.ylim(1.6, 2.4)
#     plt.xlim(0, 180)
    plt.grid()

    plt.savefig(output)
    plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocess ntuples')
    parser.add_argument('-i', '--input', nargs='+',
        help='Input file.'
    )
    parser.add_argument('-o', '--outputdir',
        default='/tmp', help='Output file. Default: (%default)s'
    )
    args = parser.parse_args()
#     print args.input

    for fn in args.input:
        try:
            outpath = os.path.join(args.outputdir, os.path.basename(fn.replace('.log', '.pdf')))
            losses = read_file(fn)
            plot(*losses, output=outpath)
        except:
            print traceback.format_exc()
