from tensorflow.keras.layers import (
    Layer,
    Conv2D,
    UpSampling2D,
    AveragePooling2D,
    LeakyReLU,
    Input,
    concatenate,
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model

class UNet_layer(Layer):
    def __init__(self, input_shape, num_classes=6, num_features=32, blocks=[1, 2, 4, 8, 16], trainable=False, **kwargs):
        super(UNet_layer, self).__init__(**kwargs)

        # Save parameters
        self.input_shape_ = input_shape
        self.num_classes = num_classes
        self.num_features = num_features
        self.blocks = blocks
        self.trainable = trainable

        # Define down-sampling layers
        self.down_stack = []
        for i, block in enumerate(self.blocks):
            self.down_stack.append(
                [
                    Conv2D(self.num_features * block, 3, strides=1, padding="same"),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.1),
                    Conv2D(self.num_features * block, 3, strides=1, padding="same"),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.1, name=f"down_block_relu_{i}"),
                    AveragePooling2D(pool_size=(2, 2), strides=2, padding="same") if i < len(self.blocks) - 1 else None,
                ]
            )

        # Define up-sampling layers
        self.up_stack = []
        for i, block in enumerate(self.blocks[:-1][::-1]):
            self.up_stack.append(
                [
                    UpSampling2D(size=(2, 2)),
                    Conv2D(self.num_features * block, 3, strides=1, padding="same"),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.1),
                    Conv2D(self.num_features * block, 3, strides=1, padding="same"),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.1),
                ]
            )

    def call(self, inputs, training=None):
        """
        Forward pass through the U-Net.
        """
        # Down-sampling path
        x = inputs
        skip_connections = []
        for i, block_layers in enumerate(self.down_stack):
            for layer in block_layers[:-1]:  # Skip the pooling layer in the last block
                x = layer(x, training=training)
            if block_layers[-1] is not None:  # Add skip connection and apply pooling if not the last block
                skip_connections.append(x)
                x = block_layers[-1](x)

        # Up-sampling path
        for i, (block_layers, skip) in enumerate(zip(self.up_stack, skip_connections[::-1])):
            x = block_layers[0](x)  # Up-sample
            x = concatenate([x, skip], axis=-1)  # Concatenate with skip connection
            for layer in block_layers[1:]:  # Apply convolutional layers
                x = layer(x, training=training)

        return x

def unet_global(image_size=(512, 512), num_classes=6, **kwargs):
    """
    Define the full U-Net model.
    """
    inputs = Input(shape=image_size + (3,))
    unet_layer_instance = UNet_layer(input_shape=image_size + (3,), **kwargs)
    extract_feature = unet_layer_instance(inputs)

    # Further processing
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(extract_feature)
    x = LeakyReLU(alpha=0.1)(x)
    x = AveragePooling2D(2, strides=2, padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = AveragePooling2D(2, strides=2, padding="same")(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(units=265, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(units=num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss = tf.keras.losses.KLDivergence()

    model.compile(loss=loss, optimizer=opt)
    return model

# Test the model
# Define input shape
image_size = (128, 128)
num_classes = 6
model = unet_global(image_size=image_size, num_classes=num_classes, num_features=32)


import numpy as np
random_input = np.random.rand(2, 128, 128, 3).astype('float32')  # Batch size 2

# Get the output
output = model.predict(random_input)
print("Output shape:", output.shape)  # Expected: (2, num_classes)


import numpy as np
import tensorflow as tf
import pandas as pd

# Simulated data generator
def simulate_data(num_samples, image_size=(128, 128), num_classes=6):
    """Generates a simulated dataset."""
    data = []
    spectrograms = {}
    eeg_specs = {}

    for i in range(num_samples):
        spec_id = f"spec{i}"
        eeg_id = f"eeg{i}"

        # Random spectrograms and EEG data
        spectrograms[spec_id] = np.random.rand(816, 400)
        eeg_specs[eeg_id] = np.random.rand(128, 256, 4)

        # Random min, max, and target
        min_val = np.random.randint(0, 500)
        max_val = min_val + np.random.randint(100, 300)
        target = np.random.randint(0, num_classes)  # Random target class

        data.append({"spec_id": spec_id, "eeg_id": eeg_id, "min": min_val, "max": max_val, "target": target})

    return pd.DataFrame(data), spectrograms, eeg_specs

# Simulated training and validation datasets
train_data, train_spectrograms, train_eeg_specs = simulate_data(num_samples=100)
val_data, val_spectrograms, val_eeg_specs = simulate_data(num_samples=20)

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, data, spectrograms, eeg_specs, batch_size=16, shuffle=True, mode="train"):
        self.data = data
        self.spectrograms = spectrograms
        self.eeg_specs = eeg_specs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.indexes = np.arange(len(self.data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.data.iloc[indexes]
        return self.__data_generation(batch_data)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_data):
        X = np.zeros((len(batch_data), 128, 128, 3), dtype=np.float32)
        y = np.zeros((len(batch_data), 6), dtype=np.float32)

        for i, row in enumerate(batch_data.itertuples()):
            spec_img = self.spectrograms[row.spec_id][:128, :128]
            eeg_img = self.eeg_specs[row.eeg_id][:128, :128, :3]
            X[i] = np.stack([spec_img, eeg_img[..., 0], eeg_img[..., 1]], axis=-1)
            y[i, row.target] = 1  # One-hot encode target

        return X, y

# Create data generators
train_generator = DataGenerator(train_data, train_spectrograms, train_eeg_specs, batch_size=16, shuffle=True)
val_generator = DataGenerator(val_data, val_spectrograms, val_eeg_specs, batch_size=16, shuffle=False)

image_size = (128, 128)
num_classes = 6
model = unet_global(image_size=image_size, num_classes=num_classes, num_features=32)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=1,
    verbose=1
)
