from __future__ import print_function
import mxnet as mx
import logging
import os
import time

def _get_lr_scheduler(args, adv=False):
    lr = args.lr
    if adv:
        lr *= args.adv_lr_scale
    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (lr, None)
    epoch_size = args.num_examples // args.batch_size
#     if 'dist' in args.kv_store:
#         epoch_size //= kv.num_workers
    begin_epoch = args.load_epoch if args.load_epoch else 0
    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d' %(lr, begin_epoch))

    steps = [epoch_size * (x-begin_epoch) for x in step_epochs if x-begin_epoch > 0]
    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor))

def _load_model(args):
    if 'load_epoch' not in args or args.load_epoch is None:
        return (None, None, None, None)
    assert args.model_prefix is not None
    model_prefix = args.model_prefix
#     symD = mx.sym.load('%s-symbol.json' % model_prefix)
    softmaxD = mx.sym.load('%s-symbol-softmax.json' % model_prefix)
    symAdv = None
#     symAdv = mx.sym.load('%s-adv-symbol.json' % model_prefix)
    param_file = '%s-%04d.params' % (model_prefix, args.load_epoch)
    adv_param_file = '%s-adv-%04d.params' % (model_prefix, args.load_epoch)
    logging.info('Load model from %s and %s', param_file, adv_param_file)
    return (softmaxD, symAdv, param_file, adv_param_file)

def _save_model(args, epoch, netD, netAdv, symD, symAdv, softmax=None):
    if args.model_prefix is None:
        return None
    dst_dir = os.path.dirname(args.model_prefix)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    model_prefix = args.model_prefix
    symD.save('%s-symbol.json' % model_prefix)
#     symAdv.save('%s-adv-symbol.json' % model_prefix)
    if softmax:
        softmax.save('%s-symbol-softmax.json' % model_prefix)

    param_name = '%s-%04d.params' % (model_prefix, epoch)
    netD.save_params(param_name)
    logging.info('Saving model parameter to %s' % param_name)

    adv_param_name = '%s-adv-%04d.params' % (model_prefix, epoch)
    netAdv.save_params(adv_param_name)
    logging.info('Saving adversarial net parameter to %s' % adv_param_name)

def _get_adversarial_weight(args, epoch=None, batch=None):
    if epoch is None or epoch >= args.adv_warmup_epochs:
        return float(args.adv_max_weight)
    else:
        wgt = float(args.adv_max_weight) / args.adv_warmup_epochs * (epoch + 1)
        if batch is None or batch >= args.adv_warmup_batches:
            return wgt
        else:
            return wgt / args.adv_warmup_batches * batch

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    train = parser.add_argument_group('Training', 'model training')
    train.add_argument('--network', type=str,
                       help='the neural network to use')
    train.add_argument('--num-layers', type=int,
                       help='number of layers in the neural network, required by some networks such as resnet')
    train.add_argument('--gpus', type=str, default='0',
                       help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    train.add_argument('--gpus-work-load', type=str, default=None,
                       help='list of gpus workload')
    train.add_argument('--kv-store', type=str, default='device',
                       help='key-value store type')
    train.add_argument('--num-epochs', type=int, default=500,
                       help='max num of epochs')
    train.add_argument('--lr', type=float, default=0.1,
                       help='initial learning rate')
    train.add_argument('--lr-factor', type=float, default=0.1,
                       help='the ratio to reduce lr on each step')
    train.add_argument('--lr-step-epochs', type=str,
                       help='the epochs to reduce the lr, e.g. 30,60')
    train.add_argument('--optimizer', type=str, default='sgd',
                       help='the optimizer type')
    train.add_argument('--mom', type=float, default=0.9,
                       help='momentum for sgd')
    train.add_argument('--wd', type=float, default=0.0001,
                       help='weight decay for sgd')
    train.add_argument('--batch-size', type=int, default=128,
                       help='the batch size')
    train.add_argument('--disp-batches', type=int, default=20,
                       help='show progress for every n batches')
    train.add_argument('--model-prefix', type=str,
                       help='model prefix')
    parser.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    train.add_argument('--load-epoch', type=int,
                       help='load the model on an epoch using the model-load-prefix')
    train.add_argument('--top-k', type=int, default=0,
                       help='report the top-k accuracy. 0 means no report.')
    train.add_argument('--test-io', action='store_true', default=False,
                       help='test reading speed without training')
    train.add_argument('--make-plots', action='store_true', default=False,
                       help='make control plots wihtout training')
    train.add_argument('--predict', action='store_true', default=False,
                       help='run prediction instead of training')
    train.add_argument('--predict-output', type=str,
                       help='predict output')
    train.add_argument('--adv-max-weight', type=float, default=50.,
                       help='max weight of adversarial loss')
    train.add_argument('--adv-warmup-epochs', type=int, default=1,
                       help='num. epochs taken to reach max weight for the advesarial loss')
    train.add_argument('--adv-warmup-batches', type=int, default=100,
                       help='num. batches taken to reach max weight for the advesarial loss')
    train.add_argument('--adv-qcd-start-label', type=int, default=11,
                       help='qcd start label')
    train.add_argument('--adv-lr-scale', type=float, default=1.,  # lr=0.001 seems good
                       help='ratio of adv. lr to classifier lr')
    train.add_argument('--adv-mass-max', type=float, default=250.,
                       help='max fatjet mass')
    train.add_argument('--adv-mass-nbins', type=int, default=50,
                       help='nbins for fatjet mass')
    train.add_argument('--adv-train-interval', type=int, default=100,
                       help='adv-to-classifier training times ratio')
    train.add_argument('--clip-gradient', type=float, default=None,
                       help='grad clipping')
    return train

class dummyKV:
    def __init__(self):
        self.rank = 0

def fit(args, symbol, data_loader, **kwargs):
    """
    train a model
    args : argparse returns
    network : the symbol definition of the nerual network
    data_loader : function that returns the train and val data iterators
    """

    # devices for training
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]
    if len(devs) == 1:
        devs = devs[0]

    # logging
    head = '%(asctime)-15s Node[0] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    # data iterators
    (train, val) = data_loader(args)
    if args.test_io:
        for i_epoch in range(args.num_epochs):
            train.reset()
            tic = time.time()
            for i, batch in enumerate(train):
                for j in batch.data:
                    j.wait_to_read()
                if (i + 1) % args.disp_batches == 0:
                    logging.info('Epoch [%d]/Batch [%d]\tSpeed: %.2f samples/sec' % (
                        i_epoch, i, args.disp_batches * args.batch_size / (time.time() - tic)))
                    tic = time.time()

        return

    if args.make_plots:
        import numpy as np
        from common.util import to_categorical, plotHist
        X_pieces = []
        y_pieces = []
        tic = time.time()
        for i, batch in enumerate(train):
            for data, label in zip(batch.data, batch.label):
                X_pieces.append(data[0].asnumpy())
                y_pieces.append(label[0].asnumpy())
            if (i + 1) % args.disp_batches == 0:
                logging.info('Batch [%d]\tSpeed: %.2f samples/sec' % (
                    i, args.disp_batches * args.batch_size / (time.time() - tic)))
                tic = time.time()
        X = np.concatenate(X_pieces).reshape((-1, train.provide_data[0][1][1]))
        y_tmp = np.concatenate(y_pieces)
        y = np.zeros(len(y_tmp), dtype=np.int)
        y[y_tmp <= 3] = 1
        y[np.logical_and(y_tmp >= 4, y_tmp <= 5)] = 2
        y[np.logical_and(y_tmp >= 6, y_tmp <= 8)] = 3
        y[np.logical_and(y_tmp >= 9, y_tmp <= 10)] = 4
        y[y_tmp >= 11] = 0
#         np.clip(X, 200, 2000, out=X)
        plotHist(X, to_categorical(y), legends=['QCD', 'top', 'W', 'Z', 'Higgs'], output='plot_%s.pdf' % train.provide_data[0][0],
#                 bins=np.linspace(200, 2000, 19), range=(200, 2000),
                bins=np.linspace(0, 250, 11),
#                 bins=np.linspace(-2.4, 2.4, 49),
                histtype='step')
        return

    logging.info('Data shape:\n' + str(train.provide_data))
    logging.info('Label shape:\n' + str(train.provide_label))

    # load model
    netD, netAdv, symD, symAdv, symSoftmax = symbol.get_net(train._data_format.num_classes, use_softmax=True, **vars(args))

    # load existing model
    _softmaxD, _symAdv, _param_file, _adv_param_file = _load_model(args)
    if _softmaxD is not None:
        assert symSoftmax.tojson() == _softmaxD.tojson()
#         assert symAdv.tojson() == _symAdv.tojson()
        netD.load_params(_param_file, ctx=devs)
        netAdv.load_params(_adv_param_file, ctx=devs)
    else:
        # init
        netD.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), ctx=devs)
        netAdv.collect_params().initialize(mx.init.Normal(0.02), ctx=devs)
        logging.debug('-' * 50)
        logging.debug(netD.collect_params())
        logging.debug('-' * 50)
        logging.debug(netAdv.collect_params())

    # loss
    lossD, lossAdv = symbol.get_loss(**vars(args))  # TODO

    # trainer
    # learning rate
    lr, lr_scheduler = _get_lr_scheduler(args)
    optimizer_params = {
            'learning_rate': lr,
            'lr_scheduler': lr_scheduler}
    if args.optimizer == 'sgd':
        optimizer_params['momentum'] = args.mom
        optimizer_params['wd'] = args.wd
    if args.clip_gradient is not None:
        optimizer_params['clip_gradient'] = args.clip_gradient
    trainerD = mx.gluon.Trainer(netD.collect_params(), args.optimizer, optimizer_params)

    # adv. trainer
    lr_adv, lr_scheduler_adv = _get_lr_scheduler(args, adv=True)
    optimizer_params_adv = {
            'learning_rate': lr_adv,
            'lr_scheduler': lr_scheduler_adv
            }
    if args.optimizer == 'sgd':
        optimizer_params_adv['momentum'] = args.mom
        optimizer_params_adv['wd'] = args.wd
    if args.clip_gradient is not None:
        optimizer_params_adv['clip_gradient'] = args.clip_gradient
    trainerAdv = mx.gluon.Trainer(netAdv.collect_params(), args.optimizer, optimizer_params_adv)

    # evaluation metric
    eval_metrics = ['accuracy', 'ce']
    if args.top_k > 0:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=args.top_k))
    if not isinstance(eval_metrics, mx.metric.EvalMetric):
        eval_metric = mx.metric.create(eval_metrics)

    eval_metric_adv = mx.metric.create(['accuracy', 'ce'])

    # callbacks that run after each batch
    batch_end_callback = [mx.callback.Speedometer(args.batch_size, args.disp_batches, auto_reset=True)]
    if 'batch_end_callback' in kwargs:
        cbs = kwargs['batch_end_callback']
        batch_end_callback += cbs if isinstance(cbs, list) else [cbs]

    eval_batch_end_callback = [mx.callback.Speedometer(args.batch_size, args.disp_batches * 10, False)]

    # save model
    save_model = False if args.dryrun or args.model_prefix is None else True

    # extra label var
    mass_label_name = 'label_%s' % train._data_format.extra_label_vars[0]
    ################################################################################
    # training loop
    ################################################################################
    train_data, eval_data = train, val
    for epoch in range(args.num_epochs):
        if epoch <= args.load_epoch:
            continue

        tic = time.time()
        eval_metric.reset()
        eval_metric_adv.reset()
        nbatch = 0
        data_iter = iter(train_data)
        end_of_batch = False
        next_data_batch = next(data_iter)
        while not end_of_batch:
            data_batch = next_data_batch
            # prepare data
            data = [data_batch.data[idx].as_in_context(devs) for idx, meta in enumerate(train.provide_data)]
            _labels = {meta[0]:data_batch.label[idx] for idx, meta in enumerate(train.provide_label)}
            label = _labels['softmax_label'].as_in_context(devs)
#             nuis = 0.01 * mx.nd.concat(*[_labels[key].reshape((-1, 1)) for key in _labels if key != 'softmax_label'], dim=1).as_in_context(devs)
            nuis = mx.nd.round(mx.nd.clip(_labels[mass_label_name].as_in_context(devs) / (args.adv_mass_max / args.adv_mass_nbins), 0, args.adv_mass_nbins - 1))

            sample_weight = None
            if args.adv_qcd_start_label is not None:
                qcds = (label >= args.adv_qcd_start_label)
#                 sample_weight = mx.nd.cast(qcds, data[0].dtype) / mx.nd.sum(qcds) * (0.2 * args.batch_size) + 1.e-6
                sample_weight = mx.nd.cast(qcds, data[0].dtype) + 1.e-6

            # train classifier w/ penalty
            if nbatch % args.adv_train_interval == 0:
                # adv wgt
                wgtAdv = _get_adversarial_weight(args, epoch, nbatch)

                with mx.autograd.record():
                    outD = netD(*data)
                    errD = lossD(outD, label)
                    if nbatch % (2 * args.adv_train_interval) == 0:
                        errD.backward()
                    else:
                        with mx.autograd.predict_mode():
                            outAdv = netAdv(outD)
                        errAdv = lossAdv(outAdv, nuis, sample_weight)
                        err = errD - wgtAdv * errAdv
                        err.backward()
    #             if args.adv_max_grad is not None:
    #                 grads = [i.grad(devs) for i in netD.collect_params().values() if i.grad_req != 'null']
    #                 # Here gradient is for the whole batch.
    #                 # So we multiply max_norm by batch_size and bptt size to balance it.
    #                 mx.gluon.utils.clip_global_norm(grads, args.adv_max_grad * args.batch_size)
                trainerD.step(data[0].shape[0])

            # train MDN
            with mx.autograd.predict_mode():
                predD = netD(*data)
            with mx.autograd.record():
                output = netAdv(predD.detach())
                errMDN = lossAdv(output, nuis, sample_weight)
                errMDN.backward()
#             if args.adv_max_grad is not None:
#                 grads = [i.grad(devs) for i in netAdv.collect_params().values() if i.grad_req != 'null']
#                 # Here gradient is for the whole batch.
#                 # So we multiply max_norm by batch_size and bptt size to balance it.
#                 mx.gluon.utils.clip_global_norm(grads, args.adv_max_grad * args.batch_size)
            trainerAdv.step(data[0].shape[0])

#             print('-' * 50)
#             print('epoch[%d] batch[%d]' % (epoch, nbatch))
#             def print_grad(net):
#                 import operator
#                 grad_dict = {}
#                 for i in net.collect_params().values():
#                     if i.grad_req != 'null':
#                         norm = mx.nd.norm(i.grad(devs)).asscalar()
#                         grad_dict[i.name] = norm
#                 sorted_g = sorted(grad_dict.items(), key=operator.itemgetter(1))
#                 for g in sorted_g[-15:]:
#                     print(g)
#                 print('---')
#             print_grad(netD)
#             print_grad(netAdv)
#             print('errD=%f, errAdv=%f, err=%f' % (mx.nd.mean(errD).asscalar(), mx.nd.mean(errAdv).asscalar(), mx.nd.mean(err).asscalar()))
#             print('errMDN=%f' % mx.nd.mean(errMDN).asscalar())

            # pre fetch next batch
            try:
                next_data_batch = next(data_iter)
            except StopIteration:
                end_of_batch = True

#             self.update_metric(eval_metric, data_batch.label)
            eval_metric.update_dict({'softmax_label':label}, {'softmax_label':mx.nd.exp(predD)})
            eval_metric_adv.update_dict({mass_label_name:nuis}, {mass_label_name:mx.nd.exp(output)})

            if (nbatch + 1) % args.adv_train_interval == 0:
                eval_metric_adv.reset()
#                 logging.debug('-' * 50)
#             if nbatch % 10 == 0:
#                 for name, val in eval_metric_adv.get_name_value():
#                     if name == 'accuracy':
#                         logging.debug('Batch[%d]: MDN-%s=%f', nbatch % args.adv_train_interval, name, val)

            if batch_end_callback is not None:
                batch_end_params = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                 eval_metric=eval_metric,
                                                 locals=locals())
                for callback in mx.base._as_list(batch_end_callback):
                    callback(batch_end_params)
            if nbatch > 1 and nbatch % args.disp_batches == 1:
                logging.debug('errD=%f, errAdv=%f, err=%f' % (mx.nd.mean(errD).asscalar(), mx.nd.mean(errAdv).asscalar(), mx.nd.mean(err).asscalar()))
                for name, val in eval_metric_adv.get_name_value():
                    logging.debug('MDN-%s=%f', name, val)
                logging.debug('wgtAdv=%f' % wgtAdv)

            nbatch += 1

        # one epoch of training is finished
        for name, val in eval_metric.get_name_value():
            logging.info('Epoch[%d] Train-%s=%f', epoch, name, val)
        # adversarial info
        logging.info('Epoch[%d] Train-%s=%f', epoch, 'MDN loss', mx.nd.mean(errMDN).asscalar())
        logging.info('Epoch[%d] Train-%s=%f, wgtAdv=%f', epoch, 'sum loss', mx.nd.mean(err).asscalar(), wgtAdv)
        # timing
        toc = time.time()
        logging.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))

        # epoch end callbacks, e.g., checkpoint
        if save_model:
            _save_model(args, epoch, netD, netAdv, symD, symAdv, symSoftmax)
        #----------------------------------------
        # evaluation on validation set
        if eval_data:
            eval_data.reset()
            eval_metric.reset()
            actual_num_batch = 0
            num_batch = None

            for nbatch, eval_batch in enumerate(eval_data):
                if num_batch is not None and nbatch == num_batch:
                    break

                # prepare data
                data = [eval_batch.data[idx].as_in_context(devs) for idx, meta in enumerate(eval_data.provide_data)]
                _labels = {meta[0]:eval_batch.label[idx] for idx, meta in enumerate(eval_data.provide_label)}
                label = _labels['softmax_label'].as_in_context(devs)

                # forward
                with mx.autograd.predict_mode():
                    predD = netD(*data)

#                 self.update_metric(eval_metric, eval_batch.label)
                eval_metric.update_dict({'softmax_label':label}, {'softmax_label':mx.nd.exp(predD)})

                if eval_batch_end_callback is not None:
                    batch_end_params = mx.model.BatchEndParam(epoch=epoch,
                                                     nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    for callback in mx.base._as_list(eval_batch_end_callback):
                        callback(batch_end_params)
                actual_num_batch += 1

            for name, val in eval_metric.get_name_value():
                logging.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

        # end of 1 epoch, reset the data-iter for another epoch
        train_data.reset()
    ##########################################################################################


def predict(args, symbol, data_loader, **kwargs):
    """
    predict with a trained a model
    args : argparse returns
    data_loader : function that returns the train and val data iterators
    """

    # logging
    head = '%(asctime)-15s Node[0] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    # data iterators
    data_iter = data_loader(args)

    # devices for training
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]
    if len(devs) == 1:
        devs = devs[0]

    # load model
    netD, netAdv, symD, symAdv, symSoftmax = symbol.get_net(data_iter._data_format.num_classes, use_softmax=True, **vars(args))

    _softmaxD, _symAdv, _param_file, _adv_param_file = _load_model(args)
    if _softmaxD is not None:
        assert symSoftmax.tojson() == _softmaxD.tojson()
        netD.load_params(_param_file, ctx=devs)

    # prediction loop
    preds = []
    for eval_batch in data_iter:
        # prepare data
        data = [eval_batch.data[idx].as_in_context(devs) for idx, meta in enumerate(data_iter.provide_data)]

        # forward
        with mx.autograd.predict_mode():
            predD = netD(*data)
            probs = mx.nd.exp(predD)
        preds.append(probs.asnumpy())

    import numpy as np
    preds = np.concatenate(preds)
    truths = data_iter.get_truths()
    observers = data_iter.get_observers()

    print(preds.shape, truths.shape, observers.shape)

    pred_output = {}
    for i, label in enumerate(data_iter._data_format.class_labels):
        pred_output['class_%s' % label] = truths[:, i]
        pred_output['score_%s' % label] = preds[:, i]
    for i, obs in enumerate(data_iter._data_format.obs_vars):
        pred_output[obs] = observers[:, i]

    import pandas as pd
    df = pd.DataFrame(pred_output)
    if args.predict_output:
        logging.info('Write prediction file to %s' % args.predict_output)
        outdir = os.path.dirname(args.predict_output)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        df.to_hdf(args.predict_output, 'Events', format='table')

        from common.util import plotROC
        plotROC(preds, truths, output=os.path.join(outdir, 'roc.pdf'))

        from root_numpy import array2root
        array2root(df.to_records(index=False), filename=args.predict_output.rsplit('.', 1)[0] + '.root', treename='Events', mode='RECREATE')
