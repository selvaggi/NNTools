import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import logging


class LRFinderStoppingCriteria():

    def __init__(self, smoothing=0.3, min_iter=20):
        """
        :param smoothing: applied to running mean which is used for thresholding (float)
        :param min_iter: minimum number of iterations before early stopping can occur (int)
        """
        self.smoothing = smoothing
        self.min_iter = min_iter
        self.first_loss = None
        self.running_mean = None
        self.counter = 0

    def __call__(self, loss):
        """
        :param loss: from single iteration (float)
        :return: indicator to stop (boolean)
        """
        self.counter += 1
        if self.first_loss is None:
            self.first_loss = loss
        if self.running_mean is None:
            self.running_mean = loss
        else:
            self.running_mean = ((1 - self.smoothing) * loss) + (self.smoothing * self.running_mean)
        return (self.running_mean > self.first_loss * 2) and (self.counter >= self.min_iter)


class LRFinder():

    def __init__(self, module):
        """
        :param learner: able to take single iteration with given learning rate and return loss
           and save and load parameters of the network (Learner)
        """
        self.module = module

    def find(self, train_data, initializer, optimizer, optimizer_params,
             lr_start=1e-6, lr_multiplier=1.1, smoothing=0.3):
        """
        :param lr_start: learning rate to start search (float)
        :param lr_multiplier: factor the learning rate is multiplied by at each step of search (float)
        :param smoothing: amount of smoothing applied to loss for stopping criteria (float)
        :return: learning rate and loss pairs (list of (float, float) tuples)
        """
        # Used to initialize weights; pass data, but don't take step.
        # Would expect for new model with lazy weight initialization
        self.module.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label, for_training=True)
        self.module.init_params(initializer=initializer, force_init=True)
        self.module.init_optimizer(optimizer=optimizer, optimizer_params=optimizer_params)

        data_iter = iter(train_data)
        end_of_batch = False
        next_data_batch = next(data_iter)

        lr = lr_start
        self.results = []  # List of (lr, loss) tuples
        stopping_criteria = LRFinderStoppingCriteria(smoothing)
        nbatch = 0
        while not end_of_batch:
            self.module._optimizer.set_learning_rate(lr)

            # Run iteration, and block until loss is calculated.
            data_batch = next_data_batch
            self.module.forward(data_batch, is_train=True)
            output = self.module.get_outputs()[0].asnumpy()
            self.module.backward()
            self.module.update()

            try:
                # pre fetch next batch
                next_data_batch = next(data_iter)
                self.module.prepare(next_data_batch, sparse_row_id_fn=None)
            except StopIteration:
                end_of_batch = True

            loss = log_loss(data_batch.label[0].asnumpy(), output, labels=range(output.shape[1]))
            self.results.append((lr, loss))
            if nbatch % 10 == 0:
                logging.info('Batch [%d]  lr=%e  cross-entropy=%f' % (nbatch, self.module._optimizer.lr, loss))
            if stopping_criteria(loss):
                break
            lr = lr * lr_multiplier
            nbatch += 1
        return self.results

    def plot(self, outputname='lr_finder.pdf'):
        lrs = [e[0] for e in self.results]
        losses = [e[1] for e in self.results]
        plt.figure(figsize=(6, 8))
        plt.scatter(lrs, losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.xscale('log')
        plt.yscale('log')
        axes = plt.gca()
        axes.set_xlim([lrs[0], lrs[-1]])
        y_lower = min(losses) * 0.8
        y_upper = losses[0] * 4
        axes.set_ylim([y_lower, y_upper])
        plt.show()
        plt.savefig(outputname)

