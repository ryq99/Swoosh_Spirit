from utils.backbone import cspdarknet53_tiny
from utils.block import conv_block, upsample, route_group

def YOLOv4_tiny(inputs, num_bbox=3, NUM_CLASS):
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

def decode_train():

def decode_tf():

def decode_tflite():

def decode_trt():

def filter_boxes():

def compute_loss():
    """

    """
    return giou_loss, conf_loss, prob_loss

