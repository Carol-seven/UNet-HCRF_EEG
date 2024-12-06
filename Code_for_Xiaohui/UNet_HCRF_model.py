import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, AveragePooling2D, UpSampling2D, \
  BatchNormalization, Activation, LeakyReLU, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import backend as K

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utility import InstanceNormalization, UNet_layer
from tensorflow.keras.layers import GroupNormalization
from tensorflow.keras.optimizers import Adam, Adadelta, SGD
import Modules.LossFunction as lf
import Modules.Common_modules as cm
import numpy as np


def unet_global(image_size = (512, 512), num_classes=6, **kwargs):
    '''
    You can adjust the Unet structure by setting:
    - kwargs.num_classes
    - kwargs.num_features
    '''
    inputs = Input(image_size + (3,))
    extract_feature = UNet_layer(**kwargs)(inputs)
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = LeakyReLU(alpha=1e-1)(x)
    x = AveragePooling2D(2, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(x)
    x = LeakyReLU(alpha=1e-1)(x)
    x = AveragePooling2D(2, strides=2, padding="same")(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(units=265, activation="relu")(x)
    
    x = Dropout(0.3)(x)
    outputs = Dense(units=num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    loss = tf.keras.losses.KLDivergence()

    model.compile(loss=loss, optimizer = opt) 
    return model




