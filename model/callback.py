import numpy as np
from keras import backend as K
from keras.callbacks import Callback, TensorBoard
from sklearn.metrics import roc_auc_score, average_precision_score


class Tensorboard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


class Roc(Callback):
    def __init__(self, val_gen):
        super(Roc, self).__init__()

        self.val_gen = val_gen

    def on_epoch_end(self, epoch, logs={}):
        val_roc, val_pr = calculate_roc_pr(self.model, self.val_gen)

        logs.update({'val_roc': val_roc, 'val_pr': val_pr})
        print('\rval_roc: %s - val_pr: %s' % (str(round(val_roc, 4)), str(round(val_pr, 4))), end=100 * ' ' + '\n')


def calculate_roc_pr(model, sequence, mask=-1, return_pred=False):
    y_true = sequence.y
    y_pred = model.predict_generator(sequence, use_multiprocessing=True, workers=6)

    if y_true.ndim == 1:
        val_roc = roc_auc_score(y_true, y_pred)
        val_pr = average_precision_score(y_true, y_pred)

    elif y_true.ndim == 2:
        y_true = y_true.transpose()
        y_pred = y_pred.transpose()

        unmask_idx = [np.where(y != mask)[0] for y in y_true]
        val_roc = [roc_auc_score(yt[idx], yp[idx]) for (yt, yp, idx) in zip(y_true, y_pred, unmask_idx)]
        val_pr = [average_precision_score(yt[idx], yp[idx]) for (yt, yp, idx) in zip(y_true, y_pred, unmask_idx)]

        val_roc = np.array(val_roc).mean()
        val_pr = np.array(val_pr).mean()
        y_pred = y_pred.transpose()

    else:
        raise ValueError("Unsupported output shape for auc calculation")

    if return_pred:
        return val_roc, val_pr, y_pred
    else:
        return val_roc, val_pr
