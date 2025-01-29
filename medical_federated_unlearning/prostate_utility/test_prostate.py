""" Test script to validate data retrival and preprocessing, loss, metrics and model instantiation as well as
centralized training. """
import tensorflow as tf
import prostate_utility as utility
from unet import UNet
from prostate_utility import JointLoss, DiceLoss

# dataset
def element_norm_fn(image, label):
    """Utility function to normalize input images."""
    return tf.cast(image, tf.float32) / 255.0, tf.expand_dims(label, axis=0)


sites = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']
# sites = ['BIDMC', 'HK', 'I2CVB', 'BMC', 'RUNMC', 'UCL']
# for site in range(len(sites)):
#     train_ds = utility.get_dataset_site(sites[site], 'train')
#     print(sites[site], train_ds, len(list(train_ds.as_numpy_iterator())))

train_ds = utility.get_dataset_site(sites[0], transform=None).map(element_norm_fn)
test_ds = utility.get_dataset_site(sites[0], 'test', transform=None).batch(1024)
example, label = list(train_ds.as_numpy_iterator())[0]
print("Example shape: ", tf.shape(example))
print("Example label: ", tf.shape(label))
# print(tf.math.reduce_max(example))
train_ds = train_ds.shuffle(buffer_size=2048 * 4, seed=1).batch(32)

# model
model = UNet(output_channels=2)
model.build((None, 3, 384, 384))
model.summary()
# tf.keras.utils.plot_model(model, to_file='./model.png', show_shapes=True)

pred = model.predict(tf.zeros((32, 3, 384, 384)))
# print("Shapes pred", tf.shape(pred))
# print("Shapes ", tf.shape(pred))

softmax_pred = tf.nn.softmax(pred, axis=1)
# print("Shapes softmax", softmax_pred)

ce_dice_loss = JointLoss()
dice_coeff = DiceLoss().dice_coef
model.compile(optimizer=tf.keras.optimizers.Adam(),
              # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
              loss=ce_dice_loss,
              metrics=[dice_coeff]
              # metrics=['accuracy']
              )
# train
history = model.fit(train_ds, epochs=4)
# history = model.fit(train_ds, epochs=4, validation_data=test_ds)
