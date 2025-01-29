""" Code adapted from https://github.com/med-air/HarmoFL/blob/63182fe8425f84f6720007386b4b6bc8e1e7e8b8/utils/dataset
.py#L41 """
import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import SimpleITK as sitk
import tensorflow as tf

def convert_from_nii_to_png(img):
    high = np.quantile(img, 0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg = (newimg * 255).astype(np.uint8)
    return newimg

def get_dataset_site(site, split='train', transform=None):
    channels = {'BIDMC': 3, 'HK': 3, 'I2CVB': 3, 'ISBI': 3, 'ISBI_1.5': 3, 'UCL': 3}
    assert site in list(channels.keys())

    base_path = './prostate/data'

    images, labels = [], []
    sitedir = os.path.join(base_path, site)

    ossitedir = np.load(base_path+"/{}-dir.npy".format(site)).tolist()

    for sample in ossitedir:
        sampledir = os.path.join(sitedir, sample)
        if os.path.getsize(sampledir) < 1024 * 1024 and sampledir.endswith("nii.gz"):
            imgdir = os.path.join(sitedir, sample[:6] + ".nii.gz")
            label_v = sitk.ReadImage(sampledir)
            image_v = sitk.ReadImage(imgdir)
            label_v = sitk.GetArrayFromImage(label_v)
            label_v[label_v > 1] = 1
            image_v = sitk.GetArrayFromImage(image_v)
            image_v = convert_from_nii_to_png(image_v)

            for i in range(1, label_v.shape[0] - 1):
                label = np.array(label_v[i, :, :])
                if np.all(label == 0):
                    continue
                image = np.array(image_v[i - 1:i + 2, :, :])
                image = np.transpose(image, (1, 2, 0))

                labels.append(label)
                images.append(image)

    labels = np.array(labels).astype(int)
    images = np.array(images)

    index = np.load(base_path+"/{}-index.npy".format(site)).tolist()

    labels = labels[index]
    images = images[index]

    #fedfa
    trainlen = int(0.6 * len(labels))
    vallen = int(0.2 * len(labels))
    testlen = int(0.2 * len(labels))

    # harmofl
    # trainlen = 0.8 * len(labels) * 0.8
    # vallen = 0.8 * len(labels) - trainlen
    # testlen = 0.2 * len(labels)

    # train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    # test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
    def transpose_fn(image, label):
        """Utility function to normalize input images."""
        image = tf.transpose(image, (2, 0, 1))
        return image, label

    if split == 'train':
        # images, labels = images[:int(trainlen)], labels[:int(trainlen)]
        labels = labels[:int(trainlen)]
        l = labels.astype(np.long).squeeze()
        img_tensor = tf.constant(images[:int(trainlen)])
        ds = tf.data.Dataset.from_tensor_slices((img_tensor, tf.constant(l)))
        ds = ds.map(transpose_fn)

    elif split == 'val':
        # images, labels = images[int(trainlen):int(trainlen + vallen)], labels[int(trainlen):int(
        #     trainlen + vallen)]
        labels = labels[int(trainlen):int(trainlen + vallen)]
        l = labels.astype(np.long).squeeze()
        img_tensor = tf.constant(images[int(trainlen):int(trainlen + vallen)])
        ds = tf.data.Dataset.from_tensor_slices((img_tensor, tf.constant(l)))
        ds = ds.map(transpose_fn)
    else:
        # images, labels = images[int(trainlen + vallen):], labels[int(trainlen + vallen):]
        labels = labels[int(trainlen + vallen):]
        l = labels.astype(np.long).squeeze()
        img_tensor = tf.constant(images[int(trainlen + vallen):])
        ds = tf.data.Dataset.from_tensor_slices((img_tensor, tf.constant(l)))
        ds = ds.map(transpose_fn)

    if transform is not None:
        return ds.map(transform)

    return ds


def element_norm_fn(image, label):
    """Utility function to normalize input images."""
    return tf.cast(image, tf.float32) / 255.0, tf.expand_dims(label, axis=0)


class RandomRotation90(tf.keras.layers.Layer):
    """ Custom keras layer to rotate the input image.
        Same as HarmoFL and FedFA papers.
        Randomly, the image is rotated by an angle in [0, 90, 180, 270].
    """
    def __init__(self, prob=1.0, seed=None):
        super(RandomRotation90, self).__init__()
        self.prob = prob
        self.seed = seed

    def call(self, input_tensor, training=True):
        if training:
            p = tf.random.uniform(shape=[], maxval=1, dtype=tf.float32, seed=self.seed)
            #print("p ", p)
            if p < self.prob:
                factor = tf.random.uniform(shape=[], maxval=4, dtype=tf.int32, seed=self.seed)
                # we use transpose here because tf.image.rot90 expects channel C on the last dimension.
                # input_tensor [C, H, W]
                transposed_image = tf.transpose(input_tensor, [1, 2, 0])
                # transposed_image [H, W, C]
                rotated_image = tf.image.rot90(transposed_image, k=factor)
                rotated_image = tf.transpose(rotated_image, [2, 0, 1])
                # rotated_image [C, H, W]
                return rotated_image
            else:
                return input_tensor
        else:
            return input_tensor


# rotate_image = tf.keras.layers.RandomRotation(0.5, seed=3)
rotate_image = RandomRotation90(seed=3)
flip_image = tf.keras.layers.RandomFlip(seed=3)

# rotate_label = tf.keras.layers.RandomRotation(0.5, seed=3)
rotate_label = RandomRotation90(seed=3)
flip_label = tf.keras.layers.RandomFlip(seed=3)


def rotate_flip_fn(image, label):
    """Utility function to preprocess input images."""
    # transform images
    rotate_flip_image = tf.keras.Sequential([
        rotate_image,
        flip_image,
    ])

    rotate_flip_label = tf.keras.Sequential([
        rotate_label,
        flip_label,
    ])

    return rotate_flip_image(image), rotate_flip_label(label)


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1.0, reduction=tf.keras.losses.Reduction.AUTO, name="dice_loss"):
        super(DiceLoss, self).__init__(reduction=reduction, name=name)
        self.smooth = smooth

    def dice_coef(self, y_true, y_pred):
        softmax_pred = tf.nn.softmax(y_pred, axis=1)
        seg_pred = tf.math.argmax(softmax_pred, axis=1)
        all_dice = 0
        y_true = tf.squeeze(y_true, axis=1)
        batch_size = tf.shape(y_true)[0]
        num_class = softmax_pred.shape[1]
        for i in range(num_class):
            # each_pred = tf.zeros(tf.shape(seg_pred))
            # each_pred[seg_pred == i] = 1

            each_pred = tf.ones(tf.shape(seg_pred), tf.float32)
            mask = tf.math.equal(seg_pred, i)
            masked = tf.cast(mask, dtype=tf.float32)
            each_pred = masked * each_pred

            # each_y_true = tf.zeros(tf.shape(y_true))
            # each_y_true[y_true == i] = 1

            each_y_true = tf.ones(tf.shape(y_true), tf.float32)
            mask = tf.math.equal(y_true, i)
            masked = tf.cast(mask, dtype=tf.float32)
            each_y_true = masked * each_y_true

            intersection = tf.reduce_sum(tf.reshape((each_pred * each_y_true), [batch_size, -1]), axis=1)

            union = tf.reduce_sum(tf.reshape(each_pred, [batch_size, -1]), axis=1) + tf.reduce_sum(
                tf.reshape(each_y_true, [batch_size, -1]), axis=1)
            dice = (2. * intersection) / (union + 1e-5)

            all_dice += tf.reduce_mean(dice)

        return all_dice * 1.0 / num_class

    def call(self, y_pred, y_true):
        softmax_pred = tf.nn.softmax(y_pred, axis=1)

        # batch_size = y_true.shape[0]
        num_class = softmax_pred.shape[1]

        # bg = tf.zeros(tf.shape(y_true))
        # bg[y_true == 0] = 1

        bg = tf.ones(tf.shape(y_true), tf.float32)
        mask = tf.math.equal(y_true, 0)
        masked = tf.cast(mask, dtype=tf.float32)
        bg = masked * bg

        # label1 = tf.zeros(tf.shape(y_true))
        # label1[y_true == 1] = 1
        label1 = tf.ones(tf.shape(y_true), tf.float32)
        mask = tf.math.equal(y_true, 1)
        masked = tf.cast(mask, dtype=tf.float32)
        label1 = masked * label1


        # label2 = tf.zeros(tf.shape(y_true))
        # label2[y_true == 2] = 1
        label2 = tf.ones(tf.shape(y_true), tf.float32)
        mask = tf.math.equal(y_true, 2)
        masked = tf.cast(mask, dtype=tf.float32)
        label2 = masked * label2

        label = tf.concat([bg, label1, label2], axis=1)

        loss = 0
        smooth = 1e-5

        for i in range(num_class):
            # slice
            # tf.print("For Shape: ", tf.shape(softmax_pred[:, i, ...]))
            # tf.print("For Shape: ", tf.shape(label[:, i, ...]))
            # softmax_pred 1, 2, 384, 384

            intersect = tf.reduce_sum(softmax_pred[:, i, ...] * label[:, i, ...])
            z_sum = tf.reduce_sum(softmax_pred[:, i, ...])
            y_sum = tf.reduce_sum(label[:, i, ...])
            loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss * 1.0 / num_class

        # Apply reduction based on the reduction mode
        if self.reduction == tf.keras.losses.Reduction.SUM:
            return tf.reduce_sum(loss)
        elif self.reduction == tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE:
            return tf.reduce_mean(loss)
        else:  # Reduction.NONE
            return loss


class JointLoss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name="joint_loss"):
        super(JointLoss, self).__init__(reduction=reduction, name=name)
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction=tf.keras.losses.Reduction.NONE)
        self.dice = DiceLoss(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        # print("joint loss tf.shape ", tf.shape(y_true))
        # y_true is [batch_size, 1, h, w] ---> [batch_size, h, w]
        # y_pred is [batch_size, num_classes, h, w].

        # transposing labels and logits
        # because TF checks last dimension (all dimension equal except for last one)
        # pred_tf_transpose = tf.transpose(y_pred, [0, 2, 3, 1])
        # gt_tf_sq = tf.squeeze(y_true, axis=1)
        # gt_tf_transpose = tf.transpose(gt_tf_sq, [1, 2])
        # ce_loss = self.ce(gt_tf_sq, pred_tf_transpose)
        # # ce_loss = self.ce(tf.squeeze(y_true, axis=1), y_pred)
        #
        # return (ce_loss + self.dice(y_pred, y_true)) / 2

        pred_tf_transpose = tf.transpose(y_pred, [0, 2, 3, 1])
        gt_tf_sq = tf.squeeze(y_true, axis=1)

        ce_loss = self.ce(gt_tf_sq, pred_tf_transpose)
        dice_loss = self.dice(y_pred, y_true)

        total_loss = (ce_loss + dice_loss) / 2      ## [batch_size, 384, 384]
        # print("Total loss: {}".format(total_loss))
        # print("MEAN --------------", tf.reduce_mean(total_loss))
        # tf.print("MEAN --------------", tf.reduce_mean(total_loss))
        # Apply reduction based on the reduction mode
        if self.reduction == tf.keras.losses.Reduction.SUM:
            # print("Reduction SUM")
            return tf.reduce_sum(total_loss)
        elif self.reduction == tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE:
            # print("Reduction OVER")
            return tf.reduce_mean(total_loss)
        else:  # Reduction.NONE
            # print("Reduction NONE")
            return tf.reduce_mean(total_loss, axis=(1,2))
            return total_loss

def dice_metric(y_true, y_pred):
        softmax_pred = tf.nn.softmax(y_pred, axis=1)
        seg_pred = tf.math.argmax(softmax_pred, axis=1)
        all_dice = 0
        y_true = tf.squeeze(y_true, axis=1)
        batch_size = tf.shape(y_true)[0]
        num_class = softmax_pred.shape[1]
        for i in range(num_class):
            # each_pred = tf.zeros(tf.shape(seg_pred))
            # each_pred[seg_pred == i] = 1

            each_pred = tf.ones(tf.shape(seg_pred), tf.float32)
            mask = tf.math.equal(seg_pred, i)
            masked = tf.cast(mask, dtype=tf.float32)
            each_pred = masked * each_pred

            # each_y_true = tf.zeros(tf.shape(y_true))
            # each_y_true[y_true == i] = 1

            each_y_true = tf.ones(tf.shape(y_true), tf.float32)
            mask = tf.math.equal(y_true, i)
            masked = tf.cast(mask, dtype=tf.float32)
            each_y_true = masked * each_y_true

            intersection = tf.reduce_sum(tf.reshape((each_pred * each_y_true), [batch_size, -1]), axis=1)

            union = tf.reduce_sum(tf.reshape(each_pred, [batch_size, -1]), axis=1) + tf.reduce_sum(
                tf.reshape(each_y_true, [batch_size, -1]), axis=1)
            dice = (2. * intersection) / (union + 1e-5)

            all_dice += tf.reduce_mean(dice)

        return all_dice * 1.0 / num_class
