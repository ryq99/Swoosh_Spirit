import os
import shutil
import tensorflow as tf
from utils.train_utils import load_freeze_layer, lr_scheduler, freeze_all, unfreeze_all
from utils.yolo import yolov4_tiny, decode_train, compute_loss

def main(data_config=None, train_config=None, model_config=None):
    # configure local GPU
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # load data
    dataset_train = Dataset(a, is_training=True)
    dataset_test = Dataset(a, is_training=False)

    # set up training epochs and steps
    steps_per_epoch = len(dataset_train)
    assert (train_config['epochs'] == train_config['backbone_epochs'] + train_config['model_epochs']), \
        'backbone/model epochs not set up correctly...'
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = train_config['warmup_epochs'] * steps_per_epoch
    total_steps = (train_config['backbone_epochs'] + train_config['model_epochs']) * steps_per_epoch

    # load freeze layers
    freeze_layers = load_freeze_layer(model_name=model_config['model_name'])

    # build model
    inputs = tf.keras.layers.Input(shape=[data_config['input_size'], data_config['input_size'], 3])
    feature_maps = yolov4_tiny(inputs, num_bbox=3, NUM_CLASS=model_config['num_class'])
    bbox_tensors = []
    for i, feature_map in enumerate(feature_maps):
        if i == 0:
            bbox_tensor = decode_train(feature_map,
                                       data_config['input_size'] // 16,
                                       NUM_CLASS=model_config['num_class'],
                                       STRIDES=model_config['strides'],
                                       ANCHORS=model_config['anchors'],
                                       i=i,
                                       XYSCALE=model_config['xyscale'])
        else:
            bbox_tensor = decode_train(feature_map,
                                       data_config['input_size'] // 32,
                                       NUM_CLASS=model_config['num_class'],
                                       STRIDES=model_config['strides'],
                                       ANCHORS=model_config['anchors'],
                                       i=i,
                                       XYSCALE=model_config['xyscale'])
        bbox_tensors.append(feature_map)
        bbox_tensors.append(bbox_tensor)
        model = tf.keras.Model(inputs=inputs, outputs=bbox_tensors)
        model.summary()

    if model_config['weights'] is None:
        print('Training from scratch')
    else:
        # incorporate transfer learning later
        pass

    optimizer = tf.keras.optimizers.Adam()
    if os.path.exists(train_config['logdir']):
        shutil.rmtree(train_config['logdir'])
    writer = tf.summary.create_file_writer(train_config['logdir'])

    def train_step(inputs, target):
        with tf.GradientTape() as tape:
            # get prediction
            pred_result = model(inputs, training=True)
            if global_steps.numpy() == 1:
                tf.print('pred_result shape =', tf.shape(pred_result))

            # initialize losses
            giou_loss = conf_loss = prob_loss = 0

            # optimization path and update weights
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred,
                                          conv,
                                          target[i][0],
                                          target[i][1],
                                          STRIDES=model_config['strides'],
                                          NUM_CLASS=model_config['num_class'],
                                          IOU_LOSS_THRESHOLD=model_config['iou_loss_threshold'],
                                          i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]
            total_loss = giou_loss + conf_loss + prob_loss
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # print step info
            tf.print('==> STEP {:4d}/{:4d}: lr = {:.6f}, giou_loss = {:4.2f}, conf_loss = {:4.2f}, '
                     'prob_loss = {:4.2f}, total_loss = {:4.2f}'
                     .format(global_steps, total_steps, optimizer.lr.numpy(), giou_loss, conf_loss, prob_loss, total_loss))

            # update learning rate
            global_steps.assign_add(1)
            lr = lr_scheduler(global_steps, warmup_steps, total_steps, train_config)
            optimizer.lr.assign(lr.numpy())

            # write summary data
            with writer.as_default():
                tf.summary.scalar('lr', optimizer.lr, step=global_steps)
                tf.summary.scalar('loss/total_loss', total_loss, step=global_steps)
                tf.summary.scalar('loss/giou_loss', giou_loss, step=global_steps)
                tf.summary.scalar('loss/conf_loss', conf_loss, step=global_steps)
                tf.summary.scalar('loss/prob_loss', prob_loss, step=global_steps)
            writer.flush()

    def test_step(inputs, target):
        with tf.GradientTape as tape:
            predict_result = model(inputs, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimization process
            for i in range(len(freeze_layers)):
                conv, pred = predict_result[i * 2], predict_result[i * 2 + 1]
                loss_items = compute_loss(pred,
                                          conv,
                                          target[i][0],
                                          STRIDES=model_config['strides'],
                                          NUM_CLASS=model_config['num_class'],
                                          IOU_LOSS_THRESHOLD=model_config['iou_loss_threshold'],
                                          i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]
            total_loss = giou_loss + conf_loss + prob_loss

            # print step info
            tf.print('==> TEST STEP {:4d}: giou_loss = {:4.2f}, conf_loss = {:4.2f}, prob_loss = {:4.2f}, total_loss = {:4.2f}'
                     .format(global_steps, giou_loss, conf_loss, prob_loss, total_loss))

    # training loop
    for epoch in range(train_config['epochs']):
        if epoch < train_config['backbone_epochs']:
            if not is_freeze:
                is_freeze = True
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    freeze_all(freeze)
        elif epoch >= train_config['backbone_epochs']:
            if is_freeze:
                is_freeze = False
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    unfreeze_all(freeze)
        for inputs, target in dataset_train:
            train_step(inputs, target)
        for inputs, target in dataset_test:
            test_step(inputs, target)
        model.save_weights('model_checkpoints/' + str(global_steps))


    if __name__ == '__main__':
        main(data_config=None,
             train_config=None,
             model_config=None)