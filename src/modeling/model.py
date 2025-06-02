import tensorflow as tf
import numpy as np
import random
import gc
import time
from stopit import threading_timeoutable as timeoutable
from typing import List

MESSAGE_1 = "Model building timeout"
MESSAGE_2 = "Evaluation timeout"

MAX_HIDDEN = 128 

class WildregressModel:
    def __init__(self, input_shape: list, control_level_shape=MAX_HIDDEN, q_level=3):
        self.input_shape = input_shape
        self.control = control_level_shape
        self.q_level = q_level

    @timeoutable(default=MESSAGE_1)
    def __call__(self, bot_pop, bot, setblockov, blocks):
        inputs = tf.keras.layers.Input(self.input_shape)
        dim_net = len(self.input_shape) - 1

        idx = [i for i, block in enumerate(setblockov) if any(x in block for x in blocks.net_lays)]
        in_nb = idx[0]
        in_block = blocks.__buildblock__(inputs, setblockov[in_nb], bot[in_nb])

        new_setblockov, new_bot = [], []
        for i in range(1, len(setblockov)):
            if not new_setblockov and setblockov[i] == []:
                new_setblockov.append(setblockov[i])
                new_bot.append(bot[i])
            elif setblockov[i] != []:
                new_setblockov.append(setblockov[i])
                new_bot.append(bot[i])

        if len(new_setblockov) > self.q_level:
            if not bot_pop[9]:
                bot_pop[8] = random.choice(np.arange(2, len(new_setblockov) - 1))
                bot_pop[9] = [0, len(new_setblockov)] + sorted(random.sample(range(1, len(new_setblockov) - 1), bot_pop[8]))
            tiers = bot_pop[9]

            brickblock = []
            for j in range(len(tiers) - 1):
                indata = in_block if j == 0 else concdata
                hidblock = [blocks.__buildblock__(indata, new_setblockov[i], new_bot[i])
                            for i in range(tiers[j], tiers[j+1])]
                if len(hidblock) > 1:
                    concdata = blocks.__flatconcat__(hidblock)
                    newshape = blocks.set_net.__redim__(concdata.shape[-1], dim_net + 1, sort=0)
                    concdata = tf.keras.layers.Reshape(newshape)(concdata)
                    brickblock.append(concdata)
                elif hidblock:
                    brickblock.append(hidblock[0])
            to_out = blocks.__flatconcat__(brickblock)
            out_block = blocks.__buildblockout__(to_out, bot_pop)
        elif new_setblockov:
            hidblock = [blocks.__buildblock__(in_block, sbo, bo)
                        for sbo, bo in zip(new_setblockov, new_bot)]
            to_out = blocks.__flatconcat__(hidblock)
            out_block = blocks.__buildblockout__(to_out, bot_pop)
        else:
            in_block_out = blocks.__flatconcat__([in_block])
            out_block = blocks.__buildblockout__(in_block_out, bot_pop)

        final_out = tf.keras.layers.Dense(units=blocks.neiro_out,
                                          activation=blocks.activ_out[bot_pop[7]])(blocks.__flatconcat__([in_block, out_block]))

        return tf.keras.Model(inputs, final_out)

# ==== Evaluation and Callbacks ====

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)

class GarbageCollectorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

@timeoutable(default=MESSAGE_2)
def evaluate_model(model, y_scaler, train_gen, val_gen, ep, verb, optimizer,
                   loss, channels, predict_lag, XVAL, YVAL):
    model.compile(optimizer, loss)
    time_callback = TimeHistory()
    clear_ozu = GarbageCollectorCallback()
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min',
                                                     factor=0.6, patience=1,
                                                     min_lr=1e-9, verbose=1)

    if isinstance(train_gen, tuple):
        train_X, train_y = train_gen
    else:
        train_X, train_y = train_gen, None

    if isinstance(val_gen, tuple):
        val_X, val_y = val_gen
    else:
        val_X, val_y = val_gen, None

    history = model.fit(train_X, train_y,
                        epochs=ep,
                        verbose=verb,
                        validation_data=(val_X, val_y),
                        callbacks=[time_callback, clear_ozu, reduce_lr])
    times_back = time_callback.times
    time_ep = np.mean(times_back)

    raw_pred = model.predict(XVAL)

    if raw_pred.ndim > 2:
        raw_pred = raw_pred.squeeze(-1)
    pred_val = y_scaler.inverse_transform(raw_pred.reshape(-1, 1))
    y_val_true = y_scaler.inverse_transform(YVAL)

    # Корреляции
    def correlate(a, b):
        ma, mb = a.mean(), b.mean()
        mab = (a * b).mean()
        sa, sb = a.std(), b.std()
        return (mab - ma * mb) / (sa * sb) if sa > 0 and sb > 0 else 0

    def auto_corr(channels, predict_lag, y_pred, y_true):
        corr, own_corr = [], []
        corr_steps = min(20, len(y_true) - 1)

        for lag in channels:
            corr.append([correlate(y_true[:len(y_true) - i], y_pred[i:]) for i in range(corr_steps)])
            own_corr.append([correlate(y_true[:len(y_true) - i], y_true[i:]) for i in range(corr_steps)])

        return corr, own_corr

    pred_val = pred_val.reshape(-1)
    y_val_true = YVAL.reshape(-1)

    min_len = min(len(pred_val), len(y_val_true))
    pred_val = pred_val[:min_len]
    y_val_true = y_val_true[:min_len]

    corr, own_corr = auto_corr(channels, predict_lag, pred_val, y_val_true)
    val = 100 * tf.keras.losses.MAE(np.array(corr), np.array(own_corr)).numpy() * history.history["val_loss"][-1]

    tf.keras.backend.clear_session()
    del model
    gc.collect()
    return val, time_ep
