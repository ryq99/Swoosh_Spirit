import numpy as np
import tensorflow as tf
from utils.backbone import cspdarknet53_tiny
from utils.block import conv_block, upsample, route_group
from utils.train_utils import bbox_iou, bbox_giou, bbox_ciou

def conv_block(inputs, channels, kernel_size=(3, 3), activation='relu', downsample=False, bn=True):
    if downsample:
        strides = (2, 2)
        padding = 'valid'
        inputs = tf.keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(inputs)
    else:
        strides = (1, 1)
        padding = 'same'
    inputs = tf.keras.layers.Conv2D(filters=channels,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    use_bias=not bn,
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.0005),
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                    bias_initializer=tf.constant_initializer(0.))(inputs)
    if bn:
        inputs = tf.keras.layers.BatchNormalization()(inputs)
    if activation == 'relu':
        outputs = tf.keras.activations.relu(inputs, alpha=0, threshold=0)
    if activation == 'leaky':
        outputs = tf.keras.activations.relu(inputs, alpha=0.1, threshold=0)
    if activation is None:
        pass
    return outputs


def upsample(inputs):
    outputs = tf.image.resize(inputs,
                              size=(inputs.shape[1] * 2, inputs.shape[2] * 2),
                              method=tf.image.ResizeMethod.BILINEAR)
    return outputs


def route_group(inputs, num_groups, group_id):
    convs = tf.split(inputs, num_or_size_splits=num_groups, axis=-1)
    return convs[group_id]


def cspdarknet53_tiny(inputs):
    x = conv_block(inputs, channels=32, downsample=True)
    x = conv_block(x, channels=64, downsample=True)
    #### block start #####
    x = conv_block(x, channels=64)
    route = x
    x = route_group(x, num_groups=2, group_id=1)
    x = conv_block(x, channels=32)
    route_1 = x
    x = conv_block(x, channels=32)
    x = tf.concat([x, route_1], axis=-1)
    x = conv_block(x, channels=64, kernel_size=(1, 1))
    x = tf.concat([route, x], axis=-1)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    #### block start #####
    x = conv_block(x, channels=128)
    route = x
    x = route_group(x, num_groups=2, group_id=1)
    x = conv_block(x, channels=64)
    route_1 = x
    x = conv_block(x, channels=64)
    x = tf.concat([x, route_1], axis=-1)
    x = conv_block(x, channels=128, kernel_size=(1, 1))
    x = tf.concat([route, x], axis=-1)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    #### block start #####
    x = conv_block(x, channels=256)
    route = x
    x = route_group(x, num_groups=2, group_id=1)
    x = conv_block(x, 128)
    route_1 = x
    x = conv_block(x, channels=128)
    x = tf.concat([x, route_1], axis=-1)
    x = conv_bloack(x, channels=256, kernel_size=(1, 1))
    x = tf.concat([route, x], axis=-1)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    x = conv_block(x, channels=512)
    return route_1, x


def darknet53_tiny(inputs):
    x = conv_block(inputs, channels=16, kernel_size=(3, 3))
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = conv_block(x, channels=32, kernel_size=(3, 3))
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = conv_block(x, channels=64, kernel_size=(3, 3))
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = conv_block(x, channels=128, kernel_size=(3, 3))
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = conv_block(x, channels=256, kernel_size=(3, 3))
    route_1 = x
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = conv_block(x, channels=512, kernel_size=(3, 3))
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = conv_block(x, channels=1024, kernel_size=(3, 3))

    return route_1, x


def yolov4_tiny(inputs, num_bbox=3, NUM_CLASS):
    """

    Return:
        [conv_mbbox, conv_lbbox]
    """
    route_1, x = cspdarknet53_tiny(inputs)
    x = conv_block(x, channels=256, kernel_size=(1, 1))

    conv_lobj_branch = conv_block(x, channels=512, kernel_size=(3, 3))
    conv_lbbox = conv_block(conv_lobj_branch,
                            channels=(NUM_CLASS + 4 + 1)*num_bbox,
                            kernel_size=(1, 1),
                            activation=None,
                            bn=False)

    x = conv_block(x, channels=128, kernel_size=(1, 1))
    x = upsample(x)
    x = tf.concat([x, route_1], axis=-1)

    conv_mobj_branch = conv_block(x, channels=256, kernel_size=(3, 3))
    conv_mbbox = conv_block(conv_mobj_branch,
                            channels=(NUM_CLASS + 4 + 1)*num_bbox,
                            kernel_size=(1, 1),
                            activation=None,
                            bn=False)
    return [conv_mbbox, conv_lbbox]


def decode():
    if FRAMEWORK == 'trt':
        return decode_trt(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)
    elif FRAMEWORK == 'tflite':
        return decode_tflite(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)
    else:
        return decode_tf(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)


def decode_train(feature_map, fm_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    """
    Create prediction

    Return:
        [pred_xywh, pred_conf, pred_prob]
    """
    feature_map = tf.reshape(feature_map,
                             shape=(tf.shape(feature_map)[0], fm_size, fm_size, 3, 5 + NUM_CLASS))

    # xywh, confidence, classification
    (fm_raw_dxdy,
     fm_raw_dwdh,
     fm_raw_conf,
     fm_raw_prob) = tf.split(feature_map, num_or_size_splits=(2, 2, 1, NUM_CLASS), axis=-1)

    # xywh activation
    xy_grid = tf.meshgrid(tf.range(fm_size), tf.range(fm_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(feature_map)[0], 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)
    tf.print('xy_grid shape =', tf.shape(xy_grid))
    pred_xy = ((tf.sigmoid(fm_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
    pred_wh = tf.exp(fm_raw_dwdh) * ANCHORS[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    # is_obj activation
    pred_conf = tf.sigmoid(fm_raw_conf)

    # P(obji|is_obj) activation
    pred_prob = tf.sigmoid(fm_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def decode_tf():
    """

    """
    return pred_xywh, pred_prob


def decode_tflite():
    """

    """
    return pred_xywh, pred_prob


def decode_trt():
    """

    """
    return pred_xywh, pred_prob


def filter_boxes():
    """

    """
    return (boxes, pred_conf)


def compute_loss(pred, conv, label, bboxes, STRIDES, NUM_CLASS, IOU_LOSS_THRESHOLD, i=0):
    """

    """
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES[i] * output_size
    conv = tf.reshape(conv, shape=(batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]
    # pred
    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]
    # label
    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < IOU_LOSS_THRESHOLD, tf.float32)

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
                    respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf) +
                    respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf))

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss

