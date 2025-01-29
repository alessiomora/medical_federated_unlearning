""" Implementation adapted from FedFA and HarmoFL papers """
import tensorflow as tf

# BATCH_NORM_DECAY = 0.997
BATCH_NORM_MOMENTUM = 0.1
BATCH_NORM_EPSILON = 1e-5
# LAYER_NORM_EPSILON = 1e-5
# GROUP_NORM_EPSILON = 1e-5


class Block(tf.keras.Model):
    """ Defining the encoder basic block """
    def __init__(self, features, seed=None):
        super().__init__()

        # if tf.keras.backend.image_data_format() == 'channels_last':
        #     channel_axis = 3
        # else:
        #     channel_axis = 1
        channel_axis = 1
        self.conv1 = tf.keras.layers.Conv2D(features, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                            use_bias=False,
                                            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                            data_format='channels_first'
                                            )
        self.bn1 = tf.keras.layers.BatchNormalization(axis=channel_axis, momentum=BATCH_NORM_MOMENTUM, epsilon=BATCH_NORM_EPSILON)

        self.conv2 = tf.keras.layers.Conv2D(features, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                            use_bias=False,
                                            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                            data_format='channels_first'
                                            )
        self.bn2 = tf.keras.layers.BatchNormalization(axis=channel_axis, momentum=BATCH_NORM_MOMENTUM, epsilon=BATCH_NORM_EPSILON)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.keras.layers.ReLU()(x)

        x = self.conv2(x)
        x = self.bn2(x)
        return tf.keras.layers.ReLU()(x)


class UNet(tf.keras.Model):
    def __init__(self, init_features=32, output_channels=2, seed=None):
        super().__init__()
        if seed is not None:
            tf.random.set_seed(seed)

        features = init_features
        self.encoder1 = Block(features)
        # self.encoder1 = _block(in_channels, features, name="enc1", affine=affine,
        #                        track_running_stats=track_running_stats)

        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', data_format='channels_first')

        # self.encoder2 = _block(features, features * 2, name="enc2", affine=affine,
        #                        track_running_stats=track_running_stats)
        self.encoder2 = Block(features * 2)

        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', data_format='channels_first')

        # self.encoder3 = _block(features * 2, features * 4, name="enc3", affine=affine,
        #                        track_running_stats=track_running_stats)
        self.encoder3 = Block(features * 4)

        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', data_format='channels_first')

        # self.encoder4 = _block(features * 4, features * 8, name="enc4", affine=affine,
        #                        track_running_stats=track_running_stats)
        self.encoder4 = Block(features * 8)

        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', data_format='channels_first')

        # self.bottleneck = _block(features * 8, features * 16, name="bottleneck", affine=affine,
        #                          track_running_stats=track_running_stats)
        self.bottleneck = Block(features * 16)

        # DECODER
        self.upconv4 = tf.keras.layers.Conv2DTranspose(features * 8, kernel_size=(2, 2), strides=2, padding='same', data_format='channels_first')
        self.decoder4 = Block(features * 8)

        self.upconv3 = tf.keras.layers.Conv2DTranspose(features * 4, kernel_size=(2, 2), strides=2, padding='same', data_format='channels_first')
        self.decoder3 = Block(features * 4)

        self.upconv2 = tf.keras.layers.Conv2DTranspose(features * 2, kernel_size=(2, 2), strides=2, padding='same', data_format='channels_first')
        self.decoder2 = Block(features * 2)

        self.upconv1 = tf.keras.layers.Conv2DTranspose(features, kernel_size=(2, 2), strides=2, padding='same', data_format='channels_first')
        self.decoder1 = Block(features)

        self.conv = tf.keras.layers.Conv2D(output_channels, kernel_size=(1, 1), strides=1, padding='same',
                                           # use_bias=False,
                                           kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                           data_format='channels_first'
                                           )

    def call(self, x):
        enc1 = self.encoder1(x)
        enc1_ = self.pool1(enc1)

        enc2 = self.encoder2(enc1_)
        enc2_ = self.pool2(enc2)

        enc3 = self.encoder3(enc2_)
        enc3_ = self.pool3(enc3)

        enc4 = self.encoder4(enc3_)
        enc4_ = self.pool4(enc4)

        bottleneck = self.bottleneck(enc4_)
        # tf.Tensor([1   1  24 512], shape=(4,), dtype=int32)

        dec4 = self.upconv4(bottleneck)
        # dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = tf.keras.layers.concatenate([dec4, enc4], axis=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        # dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = tf.keras.layers.concatenate([dec3, enc3], axis=1)

        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        # dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = tf.keras.layers.concatenate([dec2, enc2], axis=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        # dec1 = torch.cat((dec1, enc1), dim=1)
        dec0 = tf.keras.layers.concatenate([dec1, enc1], axis=1)
        dec0 = self.decoder1(dec0)
        dec0 = self.conv(dec0)

        return dec0
