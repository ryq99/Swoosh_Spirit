import tensorflow as tf
from utils.backbone import cspdarknet53_tiny
from utils.block import conv_block, upsample, route_group
from utils.train_utils import bbox_giou, bbox_ciou

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

    # xy, wh, is_obj, P(obji|is_obj)
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


def compute_loss(pred, feature_map, label, bboxes, STRIDES, NUM_CLASS, IOU_LOSS_THRESHOLD, i=0):
    """

    """
    fm_shape = tf.shape(feature_map)
    batch_size = fm_shape[0]
    fm_size = fm_shape[1]
    input_size = STRIDES[i] * fm_size
    feature_map = tf.reshape(feature_map,
                             shape=(batch_size, fm_size, fm_size, 3, 5 + NUM_CLASS))
    # feature map
    fm_raw_conf = feature_map[:, :, :, :, 4:5]
    fm_raw_prob = feature_map[:, :, :, :, 5:]
    # pred
    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]
    # label
    label_xywh = label[:, :, :, :, 0:4]
    label_conf = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 



    return giou_loss, conf_loss, prob_loss

