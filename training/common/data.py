from __future__ import print_function

import numpy as np
import mxnet as mx
import multiprocessing
import time
import logging
import tables
tables.set_blosc_max_threads(4)

def add_data_args(parser):
    data = parser.add_argument_group('Data', 'the input data')
    data.add_argument('--data-config', type=str, help='the python file for data format')
    data.add_argument('--data-train', type=str, help='the training data')
#     data.add_argument('--data-val', type=str, help='the validation data')
    data.add_argument('--train-val-split', type=float, help='fraction of files used for training')
    data.add_argument('--data-test', type=str, help='the test data')
    data.add_argument('--data-example', type=str, help='the example data')
    data.add_argument('--dryrun', action="store_true", default=False, help='Run over a small exmaple file.')
    data.add_argument('--data-names', type=str, help='the data names')
    data.add_argument('--label-names', type=str, help='the label names')
    data.add_argument('--weight-names', type=str, help='the training data')
#     data.add_argument('--num-classes', type=int, help='the number of classes')
    data.add_argument('--num-examples', type=int, help='the number of training examples')
    data.add_argument('--syn-data', action="store_true", default=False, help='Generate dummy data on the fly.')
    data.add_argument('--dataloader-nworkers', type=int, default=2, help='the number of threads used for data loader.')
    data.add_argument('--dataloader-qsize', type=int, default=256, help='the queue size for data loader.')
    data.add_argument('--dataloader-weight-scale', type=float, default=1, help='the weight scale for data loader.')
    data.add_argument('--dataloader-max-resample', type=int, default=10, help='max times to repeat the sampling.')
    return data

class DataFormat(object):

    def __init__(self, train_groups, train_vars, label_var, wgtvar, obs_vars=[], extra_label_vars=[], pad_params=None, pad_dest_vars=[], pad_src_var=None, pad_constant=None, pad_random_range=None, random_augment_vars=None, random_augment_scale=None, sort_by=None, point_mode=None, filename=None, plotting_mode=False, train_var_range=None):
        self.train_groups = train_groups  # list
        self.train_vars = train_vars  # dict
        self.sort_by = sort_by  # dict {v_group:{'var':x, 'descend':False}}
        self.label_var = label_var
        self.wgtvar = wgtvar  # set to None if not using weights
        self.obs_vars = obs_vars  # list
        self.extra_label_vars = extra_label_vars  # list
        self.point_mode = point_mode
        self.pad_params = pad_params
        if pad_params is None:
            self.pad_params = {v_group:{
                'vars':pad_dest_vars,  # list
                'src':pad_src_var,  # str
                'constant':pad_constant,  # float
                'random':pad_random_range,  # list or tuple of 2 floats
                } for v_group in train_groups}
        else:
            for v_group in train_groups:
                if v_group not in self.pad_params:
                    self.pad_params[v_group] = {'src':None, 'vars':[]}
        if pad_params is not None and pad_src_var is not None:
            logging.debug('Padding info:\n  ' + str(self.pad_params))
        self.random_augment_vars = random_augment_vars  # list
        self.random_augment_scale = random_augment_scale  # float
        self._set_range(plotting_mode, train_var_range)
        self._parse_file(filename)

    def _set_range(self, plotting_mode, train_var_range):
        # weight/var range
        self.WEIGHT_MIN = 0.
        self.WEIGHT_MAX = 1.
        if not plotting_mode:
            if train_var_range is None:
                self.VAR_MIN = -5.
                self.VAR_MAX = 5.
            else:
                self.VAR_MIN, self.VAR_MAX = train_var_range
        else:
            self.VAR_MIN = -1e99
            self.VAR_MAX = 1e99

    @staticmethod
    def nevts(filename, label_var='label'):
        with tables.open_file(filename) as f:
#             return getattr(f.root, f.root.__members__[0]).shape[0]
            return getattr(f.root, label_var).shape[0]

    @staticmethod
    def nwgtsum(filename, weight_vars='weight,class_weight'):
        wgt_vars = weight_vars.replace(' ', '').split(',')
        assert len(wgt_vars) > 0
        with tables.open_file(filename) as f:
            return np.sum(np.prod([getattr(f.root, w) for w in wgt_vars], axis=0))

    @staticmethod
    def num_classes(filename, label_var='label'):
        with tables.open_file(filename) as f:
            try:
                return getattr(f.root, label_var).shape[1]
            except IndexError:
                return getattr(f.root, label_var)[:].max()

    def _parse_file(self, filename):
        self.train_groups_shapes = {}
        with tables.open_file(filename) as f:
            self.num_classes = self.num_classes(filename, self.label_var)
            if getattr(f.root, self.label_var).title:
                self.class_labels = getattr(f.root, self.label_var).title.split(',')
            else:
                self.class_labels = [self.label_var]
            for v_group in self.train_groups:
                n_channels = len(self.train_vars[v_group])
                a = getattr(f.root, self.train_vars[v_group][0])
                if a.ndim == 3:
                    # (n, W, H)
                    width, height = int(a.shape[1]), int(a.shape[2])
                elif a.ndim == 2:
                    # (n, W)
                    width, height = int(a.shape[1]), 1
                elif a.ndim == 1:
                    # (n,)
                    width, height = 1, 1
                else:
                    raise RuntimeError
                self.train_groups_shapes[v_group] = (n_channels, width, height)
                if self.point_mode == 'NPC':
                    self.train_groups_shapes[v_group] = (width, n_channels)
                elif self.point_mode == 'NCP':
                    self.train_groups_shapes[v_group] = (n_channels, width)


class PyTableEnqueuer(object):
    """Builds a queue out of a data generator.
    see, e.g., https://github.com/fchollet/keras/blob/master/keras/engine/training.py
    # Arguments
        generator: a generator function which endlessly yields data
        pickle_safe: use multiprocessing if True, otherwise threading
    """

    def __init__(self, filelist, data_format, batch_size, workers=4, q_size=20, shuffle=True, predict_mode=False, fetch_size=100000, up_sample=False, weight_scale=1, max_resample=20):
        self._filelist = filelist
        self._data_format = data_format
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._predict_mode = predict_mode
        self._fetch_size = (fetch_size // batch_size + 1) * batch_size
        self._up_sample = up_sample
        self._weight_scale = weight_scale
        self._max_resample = max_resample

        self._workers = workers
        self._q_size = q_size

        self._lock = multiprocessing.Lock()
        self._counter = None  # how many processes are running
        self._threads = []
        self._stop_event = None
        self.queue = None

        self._file_indices = None
        self._idx = None  # position of the index for next file


    def data_generator_task(self, ifile):

        if self._stop_event.is_set():
            # do nothing if the queue has been stopped (e.g., due to exceptions)
            return

        with self._lock:
            # increase counter by 1
            self._counter.value += 1

        try:
            fbegin = 0

            with tables.open_file(self._filelist[ifile]) as f:
                nevts = getattr(f.root, f.root.__members__[0]).shape[0]

                while fbegin < nevts:
                    fend = fbegin + self._fetch_size

                    # --------- Read from files ----------
                    # features
                    X_fetch = {}
                    for v_group in self._data_format.train_groups:
                        pad_param = self._data_format.pad_params[v_group]
                        # update variable ordering if needed
                        if self._data_format.sort_by and self._data_format.sort_by[v_group]:
                            if pad_param['src'] is not None:
                                raise NotImplemented('Cannot do random pad and sorting at the same time now -- to be implemented')
                            ref_a = getattr(f.root, self._data_format.sort_by[v_group]['var'])[fbegin:fend]
                            len_a = getattr(f.root, self._data_format.sort_by[v_group]['length_var'])[fbegin:fend]
                            for i in range(len_a.shape[0]):
                                ref_a[i, int(len_a[i]):] = -np.inf if self._data_format.sort_by[v_group]['descend'] else np.inf
                            if ref_a.ndim != 2:
                                # shape should be (num_samples, num_particles)
                                raise NotImplemented('Cannot sort variable group %s'%v_group)
                            # https://stackoverflow.com/questions/10921893/numpy-sorting-a-multidimensional-array-by-a-multidimensional-array
                            if self._data_format.sort_by[v_group]['descend']:
                                sorting_indices = np.argsort(-ref_a, axis=1)
                            else:
                                sorting_indices = np.argsort(ref_a, axis=1)
                            X_group = [getattr(f.root, v_name)[fbegin:fend][np.arange(ref_a.shape[0])[:, np.newaxis], sorting_indices]
                                       for v_name in self._data_format.train_vars[v_group]]
                        else:
                            X_group = []
                            pad_mask_a = None if pad_param['src'] is None else getattr(f.root, pad_param['src'])[fbegin:fend] == 0
                            for v_name in self._data_format.train_vars[v_group]:
                                a = getattr(f.root, v_name)[fbegin:fend]
                                if v_name in pad_param['vars']:
                                    if pad_mask_a is None:
                                        raise RuntimeError('Padding `src` is not set for group %s!' % v_group)
                                    if pad_param.get('constant', None) is not None:
                                        a[pad_mask_a] = pad_param['constant']
                                    elif pad_param.get('random', None) is not None:
                                        a_rand = np.random.uniform(low=pad_param['random'][0], high=pad_param['random'][1], size=a.shape)
                                        a[pad_mask_a] = a_rand[pad_mask_a]
                                    else:
                                        raise RuntimeError('Neither `constant` nor `random` is set for padding!')
                                if not self._predict_mode and self._data_format.random_augment_vars is not None and v_name in self._data_format.random_augment_vars:
                                    a *= np.random.normal(loc=1, scale=self._data_format.random_augment_scale, size=a.shape)
                                X_group.append(a)
                        
                        shape = (-1,) + self._data_format.train_groups_shapes[v_group]  # (n, C, W, H), use -1 because end can go out of range
                        if X_group[0].ndim == 3:
                            # shape=(n, W, H): e.g., 2D image
                            assert len(X_group) == 1
                            x_arr = X_group[0]
                        elif X_group[0].ndim < 3:
                            # shape=(n, W) if ndim=2: (e.g., track list)
                            # shape=(n,) if ndim=1: (glovar var)
                            if self._data_format.point_mode == 'NPC':
                                x_arr = np.stack(X_group, axis=-1)
                            else:
                                x_arr = np.stack(X_group, axis=1)
                        else:
                            raise NotImplemented
    #                         if seq_order == 'channels_last':
    #                             x_arr = x_arr.transpose((0, 2, 1))
                        X_fetch[v_group] = np.clip(x_arr, self._data_format.VAR_MIN, self._data_format.VAR_MAX).reshape(shape)
#                         logging.debug(' -- v_group=%s, fetch_array.shape=%s, reshape=%s' % (v_group, str(X_group[0].shape), str(shape)))

                    # labels
                    y_fetch = getattr(f.root, self._data_format.label_var)[fbegin:fend]

                    # observers
                    Z_fetch = None
                    if self._predict_mode:
                        Z_fetch = np.stack([getattr(f.root, v_name)[fbegin:fend] for v_name in self._data_format.obs_vars], axis=1)

                    # extra labels
                    ext_fetch = None
                    if self._data_format.extra_label_vars:
                        ext_fetch = np.stack([getattr(f.root, v_name)[fbegin:fend] for v_name in self._data_format.extra_label_vars], axis=1)

                    # weights
                    W_fetch = None
                    if not self._predict_mode and self._data_format.wgtvar:
                        w_vars = self._data_format.wgtvar.replace(' ', '').split(',')
                        wgt = getattr(f.root, w_vars[0])[fbegin:fend]
                        for idx in range(1, len(w_vars)):
                            wgt *= getattr(f.root, w_vars[idx])[fbegin:fend]
                        W_fetch = wgt

                    fbegin += self._fetch_size
                    
                    # --------- process weight, shuffle ----------
                    n_fetched = len(y_fetch)
                    # sampling the array according to the weights (require weight<1)
                    all_indices = np.arange(n_fetched)
                    keep_indices = None
                    if W_fetch is not None:
                        randwgt = np.random.uniform(low=0, high=self._weight_scale, size=n_fetched)
                        keep_flags = randwgt < W_fetch
                        if not self._up_sample:
                            keep_indices = all_indices[keep_flags]
                        else:
                            keep_indices = [all_indices[keep_flags]]
                            n_scale = n_fetched // max(1, len(keep_indices[0]))
                            if n_scale > self._max_resample:
                                if ifile == 0 and fbegin == self._fetch_size:
                                    logging.debug('n_scale=%d is larger than the max value (%d). Setting to %d' % (n_scale, self._max_resample, self._max_resample))
                                n_scale = self._max_resample
#                             print(n_scale)
                            for _ in range(n_scale - 1):
                                randwgt = np.random.uniform(size=n_fetched)
                                keep_indices.append(all_indices[randwgt < W_fetch])
                            keep_indices = np.concatenate(keep_indices)

                    # shuffle if do training
                    shuffle_indices = None
                    if self._shuffle:
                        shuffle_indices = keep_indices if keep_indices is not None else all_indices
                        np.random.shuffle(shuffle_indices)

                    if shuffle_indices is not None or keep_indices is not None:
                        indices = shuffle_indices if shuffle_indices is not None else keep_indices
                        for v_group in X_fetch:
                            X_fetch[v_group] = X_fetch[v_group][indices]
                        y_fetch = y_fetch[indices]
                        if Z_fetch is not None:
                            Z_fetch = Z_fetch[indices]
                        if ext_fetch is not None:
                            ext_fetch = ext_fetch[indices]

                    # --------- put batches into the queue ----------
                    for b in range(0, len(y_fetch), self._batch_size):
#                         delay = np.random.uniform() / 100
#                         time.sleep(delay)
                        e = b + self._batch_size
                        X_batch = {v_group:X_fetch[v_group][b:e] for v_group in X_fetch}
                        y_batch = y_fetch[b:e]
                        Z_batch = None if Z_fetch is None else Z_fetch[b:e]
                        ext_batch = None if ext_fetch is None else ext_fetch[b:e]
                        if len(y_batch) == self._batch_size:
                            self.queue.put((X_batch, y_batch, ext_batch, Z_batch))
        except Exception:
            # set stop flag if any exception occurs
            self._stop_event.set()
            raise

        with self._lock:
            # decrease counter value by 1
            self._counter.value -= 1

    def start(self):
        """Kicks off threads which add data from the generator into the queue.
        # Arguments
            workers: number of worker threads
            max_q_size: queue size (when full, threads could block on put())
            wait_time: time to sleep in-between calls to put()
        """
        logging.debug('Starting queue, file[0]=' + self._filelist[0])

        try:
            self._counter = multiprocessing.Value('i', 0)
            self._threads = []
            self._stop_event = multiprocessing.Event()
            self.queue = multiprocessing.Queue(maxsize=self._q_size)
            self._idx = 0
            self._file_indices = np.arange(len(self._filelist))
            np.random.shuffle(self._file_indices)
            self.add()
        except:
            self.stop()
            raise

    def add(self):
        '''Try adding a process if the pool is not full.'''
        def run(ifile):
            self.data_generator_task(ifile)

        if len(self._threads) >= len(self._filelist):
            # all files are processed
            return

        try:
            if self._counter.value < self._workers:
                # Reset random seed else all children processes
                # share the same seed
                np.random.seed()
                thread = multiprocessing.Process(target=run, args=(self._file_indices[self._idx],))
                thread.daemon = True
                self._threads.append(thread)
                thread.start()
                self._idx += 1
        except:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set() and sum([t.is_alive() for t in self._threads])

    def stop(self):
        """Stop running threads and wait for them to exit, if necessary.
        Should be called by the same thread which called start().
        # Arguments
            timeout: maximum time to wait on thread.join()
        """

        logging.debug('Stopping queue, file[0]=' + self._filelist[0])

        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                thread.terminate()

        if self.queue is not None:
            self.queue.close()

        self._counter = None
        self._threads = []
        self._stop_event = None
        self.queue = None
        self._file_indices = None
        self._idx = None


class DataLoader(mx.io.DataIter):
    def __init__(self, filelist, data_format, batch_size, shuffle=True, predict_mode=False, fetch_size=600000, up_sample=True, one_hot_label=False, args=None):
        self._data_format = data_format
        self._batch_size = batch_size
        self._workers = args.dataloader_nworkers
        self._q_size = args.dataloader_qsize
        self._weight_scale = args.dataloader_weight_scale
        self._max_resample = args.dataloader_max_resample
        self._predict_mode = predict_mode
        self._one_hot_label = one_hot_label
        self.args = args

        self._provide_data = []
        for v_group in self._data_format.train_groups:
            shape = (batch_size,) + self._data_format.train_groups_shapes[v_group]
            self._provide_data.append((v_group, shape))

        self._provide_label = [('softmax_label', (batch_size,))]
        for v in self._data_format.extra_label_vars:
            self._provide_label.append(('label_' + v, (batch_size,)))

        h5_samples = sum([DataFormat.nevts(filename, self._data_format.label_var) for filename in filelist])
        self.steps_per_epoch = h5_samples // batch_size

        if not self.args.syn_data:
            self.enqueuer = PyTableEnqueuer(filelist, data_format, batch_size, self._workers, self._q_size, shuffle, predict_mode, fetch_size, up_sample, weight_scale=self._weight_scale, max_resample=self._max_resample)
            self._wait_time = 0.01  # in seconds

        self.reset()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def get_truths(self):
        return np.concatenate(self._truths)

    def get_observers(self):
        return np.concatenate(self._observers)

    def __iter__(self):
        return self

    def reset(self):

        self._ibatch = 0
        self._data = None
        self._label = None
        # stores truths and observers
        if self._predict_mode:
            self._truths = []
            self._observers = []

        if not self.args.syn_data:
            self.enqueuer.stop()
            self.enqueuer.start()


    def __next__(self):
        return self.next()

    def next(self):
        self._ibatch += 1

        if self.args.syn_data:
            if self._ibatch > self.steps_per_epoch:
                raise StopIteration
            self._data = [mx.nd.array(np.random.uniform(size=shape)) for v_group, shape in self._provide_data]
            self._label = [mx.nd.array(np.random.randint(self._data_format.num_classes, size=self.batch_size))]
            for v in self._data_format.extra_label_vars:
                self._label.append(mx.nd.random_uniform(shape=self._batch_size))
            return mx.io.DataBatch(self._data, self._label, provide_data=self.provide_data, provide_label=self.provide_label, pad=0)

        generator_output = None
        while True:
            self.enqueuer.add()
            if not self.enqueuer.queue.empty():
                generator_output = self.enqueuer.queue.get()
                break
            else:
                if not self.enqueuer.is_running():
                    break
                time.sleep(self._wait_time)

        if generator_output is None:
            self.enqueuer.stop()
            raise StopIteration

        X_batch, y_batch, ext_batch, Z_batch = generator_output

        self._data = [mx.nd.array(X_batch[v_group]) for v_group in self._data_format.train_groups]
        if self._one_hot_label:
            self._label = [mx.nd.array(y_batch)]
        else:
            self._label = [mx.nd.array(np.argmax(y_batch, axis=1))]  # cannot use one-hot labelling?
        for i, v in enumerate(self._data_format.extra_label_vars):
            self._label.append(mx.nd.array(ext_batch[:, i]))
        if Z_batch is not None:
            self._truths.append(y_batch)
            self._observers.append(Z_batch)
            if self._ibatch % (self.steps_per_epoch // 50) == 0:
                logging.info('Batch %d/%d' % (self._ibatch, self.steps_per_epoch))
#         logging.info('Batch %d/%d' % (self._ibatch, self.steps_per_epoch))
#         if self._ibatch % 100 == 0 or self._ibatch > self.steps_per_epoch - 100:
#             print(self._ibatch, ': ', np.unique(self._label[0].asnumpy(), return_counts=True))
        return mx.io.DataBatch(self._data, self._label, provide_data=self.provide_data, provide_label=self.provide_label, pad=0)
