from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from layer.keras_ssd7 import build_model
from layer.keras_ssd_loss import SSDLoss
from utils.ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from utils.ssd_batch_generator import BatchGenerator
from layer.keras_layer_AnchorBoxes import AnchorBoxes


### Make predictions

# 1: Set some necessary parameters

img_height = 300 # Height of the input images
img_width = 480 # Width of the input images
img_channels = 3 # Number of color channels of the input images
n_classes = 6 # Number of classes including the background class
min_scale = 0.08 # The scaling factor for the smallest anchor boxes
max_scale = 0.96 # The scaling factor for the largest anchor boxes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
limit_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
coords = 'centroids' # Whether the box coordinates to be used should be in the 'centroids' or 'minmax' format, see documentation
normalize_coords = False # Whether or not the model is supposed to use relative coordinates that are within [0,1]

#K.set_image_dim_ordering('th')

val_dataset = BatchGenerator(images_path='./images/',
                             include_classes='all',
                             box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])

val_dataset.parse_csv(labels_path='./images/val_values.csv',
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])

predict_generator = val_dataset.generate(batch_size=4,
                                         train=False,
                                         equalize=False,
                                         brightness=False,
                                         flip=False,
                                         translate=False,
                                         scale=False,
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         gray=False,
                                         limit_boxes=True,
                                         include_thresh=0.4,
                                         diagnostics=False)

# 2: Generate samples

X, y_true, filenames = next(predict_generator)
#i = 0 # Which batch item to look at
for i in range(0,len(filenames)):
    print("Image:", filenames[i])
    print()
    #print("Ground truth boxes:\n")
    #print(y_true[i])

    # 3: Make a prediction
    model, predictor_sizes = build_model(image_size=(img_height, img_width, img_channels),
                                          n_classes=n_classes,
                                          min_scale=min_scale,
                                          max_scale=max_scale,
                                          scales=scales,
                                          aspect_ratios_global=aspect_ratios,
                                          aspect_ratios_per_layer=None,
                                          two_boxes_for_ar1=two_boxes_for_ar1,
                                          limit_boxes=limit_boxes,
                                          variances=variances,
                                          coords=coords,
                                          normalize_coords=normalize_coords)

    model.load_weights('./crack_detection_1_weights.h5')


    #model = load_model('ssd7_custom.h5')
    y_pred = model.predict(X)

    # 4: Decode the raw prediction `y_pred`

    y_pred_decoded = decode_y2(y_pred,
                               confidence_thresh=0.5,
                              iou_threshold=0.4,
                              top_k='all',
                              input_coords='centroids',
                              normalize_coords=False,
                              img_height=None,
                              img_width=None)

    #print("Decoded predictions (output format is [class_id, confidence, xmin, xmax, ymin, ymax]):\n")
    #print(y_pred_decoded[i])

    # 5: Draw the predicted boxes onto the image
    plt.show(block=True)
    plt.interactive(False)
    fig = plt.figure(figsize=(20,12))
    fig.canvas.set_window_title(filenames[i])
    plt.imshow(X[i])

    current_axis = plt.gca()

    classes = ['crack', 'crack'] # Just so we can print class names onto the image instead of IDs

    # Draw the predicted boxes in blue
    for box in y_pred_decoded[i]:
        if (box[1] >= 0.65):
            label = 'Predicted - {}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(plt.Rectangle((box[2], box[4]), box[3]-box[2], box[5]-box[4], color='blue', fill=False, linewidth=2))
            current_axis.text(box[2], box[4], label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})

    # Draw the ground truth boxes in green (omit the label for more clarity)
    # for box in y_true[i]:
    #     label = 'Actual - {}'.format(classes[int(box[0])])
    #     current_axis.add_patch(plt.Rectangle((box[1], box[3]), box[2]-box[1], box[4]-box[3], color='green', fill=False, linewidth=2))
    #     current_axis.text(box[1], box[3], label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})
    plt.show()
