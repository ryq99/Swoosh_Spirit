import tensorflow as tf
import numpy as np

def lr_scheduler(global_steps, warmup_steps, total_steps, train_config):
    if global_steps < warmup_steps:
        lr = global_step / warmup_steps * train_config['lr_init']
    else:
        lr = train_config['lr_end'] + \
             0.5 * (train_config['lr_init'] - train_config['lr_end']) * \
             (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
    return lr