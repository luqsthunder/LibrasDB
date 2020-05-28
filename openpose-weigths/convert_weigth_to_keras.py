import sys
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
from keras.layers import *
from keras.models import *
import caffe
from time import clock
import numpy as np
import os


layers_output = './layers'
caffe_model = './pose_iter_102000.caffemodel'
caffe_proto = './pose_deploy.prototxt'
KERAS_WEIGHTS_FILE = "./model_weights.h5"
KERAS_MODEL_FILE = "./model.h5"
TEST_IMAGE_PATH = './test_image.png'


caffe.set_mode_cpu()
net = caffe.Net(caffe_proto, caffe_model, caffe.TEST)

# layer names and output shapes
#net.blobs.
for layer_name, blob in net.blobs.items():
    print(layer_name, blob.data.shape)

# write out weight matrices and bias vectors
for k, v in net.params.items():
    print(k, v[0].data.shape, v[1].data.shape)
    np.save(os.path.join(layers_output, "W_{:s}.npy".format(k)), v[0].data)
    np.save(os.path.join(layers_output, "b_{:s}.npy".format(k)), v[1].data)

print("Done !")

input_ = Input((None, None, 3), name='image')

def main_block(input_, padding='same'):
    x = Conv2D(64, (3,3), strides=(1,1), padding=padding, name='conv1_1', activation='relu')(input_)
    x = Conv2D(64, (3,3), strides=(1,1), padding=padding, name='conv1_2', activation='relu')(x)
    x = MaxPool2D((2,2), strides=(2,2), name='pool1_stage1')(x)

    x = Conv2D(128, (3,3), strides=(1,1), padding=padding, name='conv2_1', activation='relu')(x)
    x = Conv2D(128, (3,3), strides=(1,1), padding=padding, name='conv2_2', activation='relu')(x)
    x = MaxPool2D((2,2), strides=(2,2), name='pool2_stage1')(x)

    x = Conv2D(256, (3,3), strides=(1,1), padding=padding, name='conv3_1', activation='relu')(x)
    x = Conv2D(256, (3,3), strides=(1,1), padding=padding, name='conv3_2', activation='relu')(x)
    x = Conv2D(256, (3,3), strides=(1,1), padding=padding, name='conv3_3', activation='relu')(x)
    x = Conv2D(256, (3,3), strides=(1,1), padding=padding, name='conv3_4', activation='relu')(x)
    x = MaxPool2D((2,2), strides=(2,2), name='pool3_stage1')(x)

    x = Conv2D(512, (3,3), strides=(1,1), padding=padding, name='conv4_1', activation='relu')(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding=padding, name='conv4_2', activation='relu')(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding=padding, name='conv4_3', activation='relu')(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding=padding, name='conv4_4', activation='relu')(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding=padding, name='conv5_1', activation='relu')(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding=padding, name='conv5_2', activation='relu')(x)
    conv5_3_CPM = Conv2D(128, (3,3), strides=(1,1), padding=padding, name='conv5_3_CPM', activation='relu')(x)
    x = Conv2D(512, (1,1), strides=(1,1), padding='valid', name='conv6_1_CPM', activation='relu')(conv5_3_CPM)
    conv6_2_CPM = Conv2D(22, (1,1), padding='valid', strides=(1,1), name='conv6_2_CPM')(x)
    return conv5_3_CPM, conv6_2_CPM

conv5_3_CPM, conv6_2_CPM = main_block(input_)

def stage_block(conv5_3_CPM, prev_stage, stage, padding='same'):
    x = concatenate([ prev_stage, conv5_3_CPM ], axis=3, name='concat_stage{}'.format(stage))
    x = Conv2D(128, (7,7), strides=(1,1), padding=padding, name='Mconv1_stage{}'.format(stage), activation='relu')(x)
    x = Conv2D(128, (7,7), strides=(1,1), padding=padding, name='Mconv2_stage{}'.format(stage), activation='relu')(x)
    x = Conv2D(128, (7,7), strides=(1,1), padding=padding, name='Mconv3_stage{}'.format(stage), activation='relu')(x)
    x = Conv2D(128, (7,7), strides=(1,1), padding=padding, name='Mconv4_stage{}'.format(stage), activation='relu')(x)
    x = Conv2D(128, (7,7), strides=(1,1), padding=padding, name='Mconv5_stage{}'.format(stage), activation='relu')(x)
    x = Conv2D(128, (1,1), strides=(1,1), padding='valid', name='Mconv6_stage{}'.format(stage), activation='relu')(x)
    x = Conv2D(22, (1,1), strides=(1,1), padding='valid', name='Mconv7_stage{}'.format(stage))(x)
    return x

prev_stage = conv6_2_CPM
for stage in range(2, 7):
    prev_stage = stage_block(conv5_3_CPM, prev_stage, stage)

x = prev_stage
model = Model(input_, x)


# Loading from files and saving to the model internally, then using "save_weights" to store into h5
for layer in model.layers:
    layer_name = layer.name
    if (os.path.exists(os.path.join(layers_output, "W_%s.npy" % layer_name))):
        print('Loading {}'.format(layer_name))
        w = np.array(np.load(os.path.join(layers_output, "W_%s.npy" % layer_name)).tolist())
        b = np.array(np.load(os.path.join(layers_output, "b_%s.npy" % layer_name)).tolist())

        w = np.transpose(w, (2, 3, 1, 0))

        layer_weights = [w, b]
        layer.set_weights(layer_weights)

model.save_weights(KERAS_MODEL_FILE)       # SAVE WEIGHTS FILE
model.save(KERAS_MODEL_FILE)               # SAVE MODEL FILE

print("Done !")


model.load_weights(KERAS_MODEL_FILE)
print('LOADED!')
