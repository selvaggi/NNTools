from __future__ import print_function
import mxnet as mx
import logging
import os
import time

def _get_lr_scheduler(args, adv=False):
    lr = args.adv_lr if adv else args.lr
    lr_factor = args.adv_lr_factor if adv and args.adv_lr_factor else args.lr_factor
    lr_step_epochs = args.adv_lr_step_epochs if adv and args.adv_lr_step_epochs else args.lr_step_epochs

    logging.info('[%slr] init-lr=%f lr_factor=%f lr_steps=%s', 'adv-' if adv else '', lr, lr_factor, lr_step_epochs)

    if lr_factor >= 1:
        return (lr, None)

    step_epochs = [int(l) for l in lr_step_epochs.split(',')]  # e.g., [20, 40, 60]
    step_lr = [lr * (lr_factor ** (n + 1)) for n in range(len(step_epochs))]
    def _get_lr(epoch):
        if not step_epochs or epoch < step_epochs[0]:
            return lr
        if epoch >= step_epochs[-1]:
            return step_lr[-1]
        for k in range(len(step_epochs) - 1):
            if epoch >= step_epochs[k] and epoch < step_epochs[k + 1]:
                return step_lr[k]
    return (lr, _get_lr)

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
#     symD.save('%s-symbol.json' % model_prefix)
#     symAdv.save('%s-adv-symbol.json' % model_prefix)
    if softmax:
        softmax.save('%s-symbol-softmax.json' % model_prefix)

    param_name = '%s-%04d.params' % (model_prefix, epoch)
#     netD.save_params(param_name)
    netD.export(model_prefix)
    os.rename('%s-0000.params' % model_prefix, param_name)
    logging.info('Saving model parameter to %s' % param_name)

    adv_param_name = '%s-adv-%04d.params' % (model_prefix, epoch)
    netAdv.save_params(adv_param_name)
    logging.info('Saving adversarial net parameter to %s' % adv_param_name)

def _load_classifier(args):
    # load model
    sym = mx.sym.load('%s-symbol.json' % args.cls_model_prefix)
    softmax = 0.1 * mx.sym.log_softmax(sym.get_internals()['fc_out_output'], name='softmax')
    inputs = [mx.sym.var(name) for name in args.data_names.split(',')]
    params = mx.gluon.ParameterDict(prefix='cls_')
    ctx = mx.cpu() if args.gpus is None or args.gpus is '' else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    net = mx.gluon.SymbolBlock(inputs=inputs, outputs=softmax, params=params)
    net.load_params('%s-%04d.params' % (args.cls_model_prefix, args.cls_load_epoch), ctx=ctx)
    net.hybridize()

    logging.info('Loaded classifier model %s_%04d.params', args.cls_model_prefix, args.cls_load_epoch)
    return net

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
    train.add_argument('--beta1', type=float, default=0.9,
                       help='beta1 for adam')
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
    train.add_argument('--predict-all', action='store_true', default=False,
                       help='run all predictions')
    train.add_argument('--predict-output', type=str,
                       help='predict output')
    train.add_argument('--cls-model-prefix', type=str,
                       help='base classifier model prefix')
    train.add_argument('--cls-load-epoch', type=int, default=0,
                       help='load the base classifier model on an epoch using the model-load-prefix')
    train.add_argument('--adv-lambda', type=float, default=10.,
                       help='weight of adversarial loss')
    train.add_argument('--adv-qcd-start-label', type=int, default=11,
                       help='qcd start label')
    train.add_argument('--adv-lr', type=float, default=0.001,  # lr=0.001 seems good
                       help='adv lr')
    train.add_argument('--adv-lr-factor', type=float, default=None,
                       help='the ratio to reduce lr on each step for adv')
    train.add_argument('--adv-lr-step-epochs', type=str, default=None,
                       help='the epochs to reduce the adv-lr, e.g. 30,60')
    train.add_argument('--adv-mass-max', type=float, default=250.,
                       help='max fatjet mass')
    train.add_argument('--adv-mass-nbins', type=int, default=50,
                       help='nbins for fatjet mass')
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
    ndevs = len(devs)

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

    # load the base classifier
    base_classifier_net = _load_classifier(args)

    # load model
    netD, netAdv, symD, symAdv, symSoftmax = symbol.get_net(train._data_format.num_classes, use_softmax=True, **vars(args))

    # load existing model
    _softmaxD, _symAdv, _param_file, _adv_param_file = _load_model(args)
    if _softmaxD is not None:
        assert symSoftmax.tojson() == _softmaxD.tojson()
#         assert symAdv.tojson() == _symAdv.tojson()
        try:
            netD.load_params(_param_file, ctx=devs)  # works with block.save_params()
        except AssertionError:
            netD.collect_params().load(_param_file, ctx=devs)  # work with block.export()
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
    lr, lr_getter = _get_lr_scheduler(args)
    optimizer_params = {'learning_rate': lr}
    if args.optimizer == 'adam':
        optimizer_params['beta1'] = args.beta1
    elif args.optimizer == 'sgd':
        optimizer_params['momentum'] = args.mom
        optimizer_params['wd'] = args.wd
    trainerD = mx.gluon.Trainer(netD.collect_params(), args.optimizer, optimizer_params)

    # adv. trainer
    lr_adv, lr_getter_adv = _get_lr_scheduler(args, adv=True)
    optimizer_params_adv = {'learning_rate': lr_adv}
    if args.optimizer == 'adam':
        optimizer_params_adv['beta1'] = args.beta1
    elif args.optimizer == 'sgd':
        optimizer_params_adv['momentum'] = args.mom
        optimizer_params_adv['wd'] = args.wd
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
    pt_label_name = 'label_%s' % train._data_format.extra_label_vars[1]
    ################################################################################
    # training loop
    ################################################################################
    train_data, eval_data = train, val
    for epoch in range(args.num_epochs):
        if epoch <= args.load_epoch:
            continue

        if lr_getter:
            trainerD.set_learning_rate(lr_getter(epoch))
        if lr_getter_adv:
            trainerAdv.set_learning_rate(lr_getter_adv(epoch))
        logging.info('Epoch[%d] lrD=%g, lrAdv=%g', epoch, trainerD.learning_rate, trainerAdv.learning_rate)

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
            _data = [mx.gluon.utils.split_and_load(data_batch.data[idx], devs) for idx, meta in enumerate(train.provide_data)]
            data = [[_data[idx][idev] for idx in range(len(train.provide_data))] for idev in range(ndevs)]
            with mx.autograd.predict_mode():
                base_preds = [base_classifier_net(*d) for d in data]

            _labels = {meta[0]:data_batch.label[idx] for idx, meta in enumerate(train.provide_label)}
            label = mx.gluon.utils.split_and_load(_labels['softmax_label'], devs)

            _mass = mx.gluon.utils.split_and_load(_labels[mass_label_name], devs)
            _pt = mx.gluon.utils.split_and_load(_labels[pt_label_name], devs)
            _rho = [2.*mx.nd.log(mx.nd.clip(m, 1, 1000) / p) for m, p in zip(_mass, _pt)]  # rho = ln(m^2/p^2)

            _extra = [mx.nd.stack(_mass[i] / 1000., mx.nd.log(_pt[i]) / 10., _rho[i] / 10., axis=1) for i in range(ndevs)]
            preds_plus_extra = [mx.nd.concat(base_preds[i], _extra[i], dim=1) for i in range(ndevs)]
#             print('-' * 50)
#             print(label[0][0])
#             print(preds_plus_extra[0][0])
#             print('-' * 50)

#             nuis = 0.01 * mx.nd.concat(*[_labels[key].reshape((-1, 1)) for key in _labels if key != 'softmax_label'], dim=1).as_in_context(devs)
#             _nuis = mx.gluon.utils.split_and_load(_labels[mass_label_name], devs)
            nuis = [mx.nd.round(mx.nd.clip(_n / (float(args.adv_mass_max) / args.adv_mass_nbins), 0, args.adv_mass_nbins - 1)) for _n in _mass]

            sample_weight = None
            sample_weight_sum = data_batch.data[0].shape[0]
            if args.adv_qcd_start_label is not None:
                sample_weight = [mx.nd.cast(l >= args.adv_qcd_start_label, data_batch.data[0].dtype) for l in label]
                sample_weight_sum = mx.nd.sum(_labels['softmax_label'] >= args.adv_qcd_start_label).asscalar()

            # from the training of the classifier
            errD = 0 
            errAdv = 0
            err = 0
            # from the training of the adversary
            errMDN = 0 

            ############################
            # (1) first train the adversary
            ############################
            with mx.autograd.record():
                outD = [netD(d) for d in preds_plus_extra]
                output = [netAdv(p.detach()) for p in outD]
                lossesR = [lossAdv(output[idev], nuis[idev], sample_weight[idev]) for idev in range(ndevs)]
            for l in lossesR:
                l.backward()
                errMDN += mx.nd.mean(l).asscalar()
            trainerAdv.step(int(sample_weight_sum))
            ############################

            ############################
            # (2) then update classifier
            ############################
            with mx.autograd.record():
                lossesD = [lossD(out, l) for out, l in zip(outD, label)]
                outAdv = [netAdv(out) for out in outD]
                lossesAdv = [lossAdv(outAdv[idev], nuis[idev], sample_weight[idev]) for idev in range(ndevs)]
                losses = [lD - args.adv_lambda * lAdv for lD, lAdv in zip(lossesD, lossesAdv)]
            for l in losses:
                l.backward()
            for idev in range(ndevs):
                errD += mx.nd.mean(lossesD[idev]).asscalar()
                errAdv += mx.nd.mean(lossesAdv[idev]).asscalar()
                err += mx.nd.mean(losses[idev]).asscalar()
            trainerD.step(data_batch.data[0].shape[0])
            ############################

            # pre fetch next batch
            try:
                next_data_batch = next(data_iter)
            except StopIteration:
                end_of_batch = True

            for idev in range(ndevs):
                eval_metric.update_dict({'softmax_label':label[idev]}, {'softmax_label':mx.nd.exp(outD[idev])})
                eval_metric_adv.update_dict({mass_label_name:nuis[idev]}, {mass_label_name:mx.nd.exp(output[idev])})

            if batch_end_callback is not None:
                batch_end_params = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                 eval_metric=eval_metric,
                                                 locals=locals())
                for callback in mx.base._as_list(batch_end_callback):
                    callback(batch_end_params)
            if nbatch > 1 and nbatch % args.disp_batches == 1:
                logging.debug('errD=%f, errAdv=%f, err=%f' % (errD / ndevs, errAdv / ndevs, err / ndevs))
                for name, val in eval_metric_adv.get_name_value():
                    logging.debug('MDN-%s=%f', name, val)
                logging.debug('wgtAdv=%f, qcdSumWgt=%f', args.adv_lambda, sample_weight_sum)

            nbatch += 1

        # one epoch of training is finished
        for name, val in eval_metric.get_name_value():
            logging.info('Epoch[%d] Train-%s=%f', epoch, name, val)
        # adversarial info
        logging.info('Epoch[%d] Train-%s=%f', epoch, 'MDN loss', errMDN / ndevs)
        logging.info('Epoch[%d] Train-%s=%f, wgtAdv=%f', epoch, 'sum loss', err / ndevs, args.adv_lambda)
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
                _data = [mx.gluon.utils.split_and_load(eval_batch.data[idx], devs) for idx, meta in enumerate(eval_data.provide_data)]
                data = [[_data[idx][idev] for idx in range(len(eval_data.provide_data))] for idev in range(ndevs)]
                with mx.autograd.predict_mode():
                    base_preds = [base_classifier_net(*d) for d in data]

                _labels = {meta[0]:eval_batch.label[idx] for idx, meta in enumerate(eval_data.provide_label)}
                label = mx.gluon.utils.split_and_load(_labels['softmax_label'], devs)

                _mass = mx.gluon.utils.split_and_load(_labels[mass_label_name], devs)
                _pt = mx.gluon.utils.split_and_load(_labels[pt_label_name], devs)
                _rho = [2.*mx.nd.log(mx.nd.clip(m, 1, 1000) / p) for m, p in zip(_mass, _pt)]  # rho = ln(m^2/p^2)

                _extra = [mx.nd.stack(_mass[i] / 1000., mx.nd.log(_pt[i]) / 10., _rho[i] / 10., axis=1) for i in range(ndevs)]
                preds_plus_extra = [mx.nd.concat(base_preds[i], _extra[i], dim=1) for i in range(ndevs)]

                # forward
                with mx.autograd.predict_mode():
                    predD = [netD(d) for d in preds_plus_extra]

#                 self.update_metric(eval_metric, eval_batch.label)
                for idev in range(ndevs):
                    eval_metric.update_dict({'softmax_label':label[idev]}, {'softmax_label':mx.nd.exp(predD[idev])})

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

    # load the base classifier
    base_classifier_net = _load_classifier(args)

    # load model
    netD, netAdv, symD, symAdv, symSoftmax = symbol.get_net(data_iter._data_format.num_classes, use_softmax=True, **vars(args))

    _softmaxD, _symAdv, _param_file, _adv_param_file = _load_model(args)
    if _softmaxD is not None:
        if symSoftmax.tojson() != _softmaxD.tojson():
            print(symSoftmax.tojson())
            print('-' * 50)
            print(_softmaxD.tojson())
            raise RuntimeError
        try:
            netD.load_params(_param_file, ctx=devs)  # works with block.save_params()
        except AssertionError:
            netD.collect_params().load(_param_file, ctx=devs)  # work with block.export()

    def _predict(data_iter, outpath):
        # prediction loop
        mass_label_name = 'label_%s' % data_iter._data_format.extra_label_vars[0]
        pt_label_name = 'label_%s' % data_iter._data_format.extra_label_vars[1]

        preds = []
        for eval_batch in data_iter:
            # prepare data
            data = [eval_batch.data[idx].as_in_context(devs) for idx, meta in enumerate(data_iter.provide_data)]
            with mx.autograd.predict_mode():
                base_preds = base_classifier_net(*data)

            _labels = {meta[0]:eval_batch.label[idx] for idx, meta in enumerate(eval_batch.provide_label)}
            _mass = _labels[mass_label_name].as_in_context(devs)
            _pt = _labels[pt_label_name].as_in_context(devs)
            _rho = 2.*mx.nd.log(mx.nd.clip(_mass, 1, 1000) / _pt)  # rho = ln(m^2/p^2)

            _extra = mx.nd.stack(_mass / 1000., mx.nd.log(_pt) / 10., _rho / 10., axis=1)
            preds_plus_extra = mx.nd.concat(base_preds, _extra, dim=1)

            # forward
            with mx.autograd.predict_mode():
                predD = netD(preds_plus_extra)
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
        if outpath:
            logging.info('Write prediction file to %s' % outpath)
            outdir = os.path.dirname(outpath)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            df.to_hdf(outpath, 'Events', format='table')

            from common.util import plotROC
            plotROC(preds, truths, output=os.path.join(outdir, 'roc.pdf'))

            from root_numpy import array2root
            array2root(df.to_records(index=False), filename=outpath.rsplit('.', 1)[0] + '.root', treename='Events', mode='RECREATE')

    if args.predict_all:
        import re
        import glob
        test_input = re.sub(r'\/JMAR.*\/.*\/', '/_INPUT_/', args.data_test)
        pred_output = re.sub(r'\/JMAR.*\/.+h5', '/_OUTPUT_', args.predict_output)
        for a in ['JMAR', 'JMAR_lowM']:
            for b in ['Top', 'W', 'Z', 'Higgs', 'QCD']:
                if a == 'JMAR_lowM' and b == 'QCD': b = 'QCD_Flat'
                args.data_test = test_input.replace('_INPUT_', '%s/%s' % (a, b))
                args.predict_output = pred_output.replace('_OUTPUT_', '%s/mx-pred_%s.h5' % (a, b))
                if len(glob.glob(args.data_test)) == 0:
                    logging.warning('No files found in %s, ignoring...', args.data_test)
                    continue
                data_iter = data_loader(args)
                _predict(data_iter, args.predict_output)
    else:
        _predict(data_iter, args.predict_output)
