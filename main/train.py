import tensorflow as tf

def main(log_dir='./data/log',
         if_freeze=False,
         steps_per_epoch=,
         data_config=None,
         train_config=None,
         model_config=None):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    train = Dataset(, is_training=True)
    test = Dataset(, is_training=False)

    #log_dir = log_dir
    #isfreeze = False

    # set up training epochs and steps
    steps_per_epoch = len(train)
    backbone_epochs = train_config['backbone_epochs']
    model_epochs = train_config['model_epochs']
    global_steps = tf.Variable(1, trainable=False, dtype=tf.in64)
    warmpup_steps = train_config['warmup_epochs'] * steps_per_epoch
    total_steps = (backbone_epochs + model_epochs) * steps_per_epoch

    inputs = tf.keras.layers.Input(shape=[data_conifg['img_width'], data_config['img_height'], 3])

    # STRIDES, ANCHORS, NUM_CLASS, XYSCALE =
    IOU_LOSS_THRESHOLD = train_config['iou_loss_threshold']

    #freeze_layers =

    feature_maps = YOLOv4_tiny(inputs, num_bbox=3, NUM_CLASS=model_conifg['num_class'])

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        if i == 0:










