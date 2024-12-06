from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, AveragePooling2D, UpSampling2D, \
  BatchNormalization, Activation, LeakyReLU, Dropout
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GroupNormalization

"""
Normalization layer that is used in:
    Chen S, Gamechi Z S, Dubost F, et al. An end-to-end approach to segmentation in medical images with CNN and posterior-CRF.
"""

class UNet_layer(Layer):
    def __init__(self, input_shape, num_classes=6, num_features=32, blocks=[1, 2, 4, 8, 16], trainable=False, **kwargs):
        self.input_shape_ = kwargs.get('input_shape' ,input_shape)
        self.num_classes = kwargs.get('num_classes' ,num_classes)
        self.num_features = kwargs.get('num_features' ,num_features) # default 32
        self.blocks = blocks # default [1, 2, 4, 8, 16]
        self.down_block_names = [f'down_block_relu_{i}' for i in range(len(blocks)-1)] + [f'down_block_relu_{len(blocks)-1}']
        inputs = Input(shape=self.input_shape_)
        x = inputs
        skips = []
        for i, block in enumerate(self.blocks):
            x = Conv2D(self.num_features * block, 3, strides=1, padding="same")(x)
            x = GroupNormalization(groups=-1)(x)
            x = LeakyReLU(alpha=1e-1)(x)
            x = Conv2D(self.num_features * block, 3, strides=1, padding="same")(x)
            x = GroupNormalization(groups=-1)(x)
            x = LeakyReLU(alpha=1e-1, name=f'down_block_relu_{i}')(x)
            if i < len(self.blocks) - 1:
                x = AveragePooling2D(2, strides=2, padding="same", name=f'down_block_pooling_{i}')(x)
        base_model = Model(inputs=inputs, outputs=x)
        self.down_stack = Model(inputs, outputs = [
            base_model.get_layer(name).output for name in self.down_block_names
        ])  # Outputs: bottleneck + skips
        self.down_stack.trainable = trainable
        super(UNet_layer, self).__init__(**kwargs)


    def call(self, inputs, training =None):
        
        skips = self.down_stack(inputs)
        x = skips[-1]
        skips = skips[:-1][::-1]
        for up_block, skip in zip(self.blocks[:-1][::-1], skips):
            x = concatenate([UpSampling2D(size=2)(x), skip])
            x = Conv2D(self.num_features * up_block, 3, strides=1, padding="same")(x)
            x = GroupNormalization(groups=-1,)(x)
            x = LeakyReLU(alpha=1e-1)(x)
            x = Conv2D(self.num_features * up_block, 3, strides=1, padding="same")(x)
            x = GroupNormalization(groups=-1,)(x)
            x = LeakyReLU(alpha=1e-1)(x)
        return x