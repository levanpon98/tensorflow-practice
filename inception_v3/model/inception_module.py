import tensorflow as tf
from tensorflow.keras import layers


class BasicConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding):
        super(BasicConv2D, self).__init__()
        self.conv = layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)

        return x


class Preprocess(tf.keras.layers.Layer):
    def __init__(self):
        super(Preprocess, self).__init__()
        self.conv1 = BasicConv2D(filters=32,
                                 kernel_size=(3, 3),
                                 strides=2,
                                 padding='same')
        self.conv2 = BasicConv2D(filters=32,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding='same')
        self.conv3 = BasicConv2D(filters=64,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding='same')
        self.maxpool1 = tf.keras.layers.MaxPool2D((3, 3), strides=2, padding='same')

        self.conv4 = BasicConv2D(filters=80,
                                 kernel_size=(1, 1),
                                 strides=1,
                                 padding='same')
        self.conv5 = BasicConv2D(filters=192,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding='same')
        self.maxpool2 = tf.keras.layers.MaxPool2D((3, 3), strides=2, padding='same')

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.maxpool1(x)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.maxpool2(x)

        return x


class InceptionModule1(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(InceptionModule1, self).__init__()

        # branch 0
        self.conv2d_b0 = BasicConv2D(filters=64,
                                     kernel_size=(1, 1),
                                     strides=1,
                                     padding='same')

        # branch 1
        self.conv2d_b1_1 = BasicConv2D(filters=48,
                                       kernel_size=(1, 1),
                                       strides=1,
                                       padding='same')
        self.conv2d_b1_2 = BasicConv2D(filters=64,
                                       kernel_size=(5, 5),
                                       strides=1,
                                       padding='same')
        # branch 2
        self.conv2d_b2_1 = BasicConv2D(filters=64,
                                       kernel_size=(1, 1),
                                       strides=1,
                                       padding='same')
        self.conv2d_b2_2 = BasicConv2D(filters=96,
                                       kernel_size=(3, 3),
                                       strides=1,
                                       padding='same')
        self.conv2d_b2_3 = BasicConv2D(filters=96,
                                       kernel_size=(3, 3),
                                       strides=1,
                                       padding='same')

        # branch 3
        self.avgpool = tf.keras.layers.AvgPool2D((3, 3), strides=1, padding='same')

        self.conv2d_b3 = BasicConv2D(filters=filters,
                                     kernel_size=(1, 1),
                                     strides=1,
                                     padding='same')

    def call(self, inputs, training=None, **kwargs):
        b0 = self.conv2d_b0(inputs, training=training)

        b1 = self.conv2d_b1_1(inputs, training=training)
        b1 = self.conv2d_b1_2(b1, training=training)

        b2 = self.conv2d_b2_1(inputs, training=training)
        b2 = self.conv2d_b2_2(b2, training=training)
        b2 = self.conv2d_b2_3(b2, training=training)

        b3 = self.avgpool(inputs)
        b3 = self.conv2d_b3(b3, training=training)

        x = tf.keras.layers.concatenate([b0, b1, b2, b3])

        return x


class InceptionModule2(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionModule2, self).__init__()

        # branch 0
        self.conv2d_b0 = BasicConv2D(filters=384,
                                     kernel_size=(3, 3),
                                     strides=2,
                                     padding='valid')
        # branch 1
        self.conv2d_b1_1 = BasicConv2D(filters=64,
                                       kernel_size=(1, 1),
                                       strides=1,
                                       padding='same')
        self.conv2d_b1_2 = BasicConv2D(filters=96,
                                       kernel_size=(3, 3),
                                       strides=1,
                                       padding='same')
        self.conv2d_b1_3 = BasicConv2D(filters=96,
                                       kernel_size=(3, 3),
                                       strides=2,
                                       padding='valid')

        # branch 2
        self.maxpool = tf.keras.layers.MaxPool2D((3, 3), strides=2, padding='valid')

    def call(self, inputs, training=None, **kwargs):
        b0 = self.conv2d_b0(inputs, training=training)

        b1 = self.conv2d_b1_1(inputs, training=training)
        b1 = self.conv2d_b1_2(b1, training=training)
        b1 = self.conv2d_b1_3(b1, training=training)

        b2 = self.maxpool(inputs)

        x = tf.keras.layers.concatenate([b0, b1, b2])

        return x


class InceptionModule3(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(InceptionModule3, self).__init__()

        # branch 0
        self.conv2d_b0 = BasicConv2D(filters=192,
                                     kernel_size=(1, 1),
                                     strides=1,
                                     padding='same')

        # branch 1
        self.conv2d_b1_1 = BasicConv2D(filters=filters,
                                       kernel_size=(1, 1),
                                       strides=1,
                                       padding='same')
        self.conv2d_b1_2 = BasicConv2D(filters=filters,
                                       kernel_size=(1, 7),
                                       strides=1,
                                       padding='same')
        self.conv2d_b1_3 = BasicConv2D(filters=192,
                                       kernel_size=(7, 1),
                                       strides=1,
                                       padding='same')

        # branch 2
        self.conv2d_b2_1 = BasicConv2D(filters=filters,
                                       kernel_size=(1, 1),
                                       strides=1,
                                       padding='same')
        self.conv2d_b2_2 = BasicConv2D(filters=filters,
                                       kernel_size=(7, 1),
                                       strides=1,
                                       padding='same')
        self.conv2d_b2_3 = BasicConv2D(filters=filters,
                                       kernel_size=(1, 7),
                                       strides=1,
                                       padding='same')
        self.conv2d_b2_4 = BasicConv2D(filters=filters,
                                       kernel_size=(7, 1),
                                       strides=1,
                                       padding='same')
        self.conv2d_b2_5 = BasicConv2D(filters=192,
                                       kernel_size=(1, 7),
                                       strides=1,
                                       padding='same')

        # branch 3
        self.avgpool = tf.keras.layers.AvgPool2D((3, 3), strides=1, padding='same')
        self.conv2d_b3 = BasicConv2D(filters=192,
                                     kernel_size=(1, 1),
                                     strides=1,
                                     padding='same')

    def call(self, inputs, training=None, **kwargs):
        b0 = self.conv2d_b0(inputs, training=training)

        b1 = self.conv2d_b1_1(inputs, training=training)
        b1 = self.conv2d_b1_2(b1, training=training)
        b1 = self.conv2d_b1_3(b1, training=training)

        b2 = self.conv2d_b2_1(inputs, training=training)
        b2 = self.conv2d_b2_2(b2, training=training)
        b2 = self.conv2d_b2_3(b2, training=training)
        b2 = self.conv2d_b2_4(b2, training=training)
        b2 = self.conv2d_b2_5(b2, training=training)

        b3 = self.avgpool(inputs)
        b3 = self.conv2d_b3(inputs, training=training)

        x = tf.keras.layers.concatenate([b0, b1, b2, b3], axis=-1)

        return x


class InceptionModule4(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionModule4, self).__init__()
        # branch 0
        self.conv_b0_1 = BasicConv2D(filters=192,
                                     kernel_size=(1, 1),
                                     strides=1,
                                     padding="same")
        self.conv_b0_2 = BasicConv2D(filters=320,
                                     kernel_size=(3, 3),
                                     strides=2,
                                     padding="valid")

        # branch 1
        self.conv_b1_1 = BasicConv2D(filters=192,
                                     kernel_size=(1, 1),
                                     strides=1,
                                     padding="same")

        self.conv_b1_2 = BasicConv2D(filters=192,
                                     kernel_size=(1, 7),
                                     strides=1,
                                     padding="same")

        self.conv_b1_3 = BasicConv2D(filters=192,
                                     kernel_size=(7, 1),
                                     strides=1,
                                     padding="same")

        self.conv_b1_4 = BasicConv2D(filters=192,
                                     kernel_size=(3, 3),
                                     strides=2,
                                     padding="valid")

        # branch 2
        self.maxpool_b2_1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                      strides=2,
                                                      padding="valid")

    def call(self, inputs, training=None, **kwargs):
        b0 = self.conv_b0_1(inputs, training=training)
        b0 = self.conv_b0_2(b0, training=training)

        b1 = self.conv_b1_1(inputs, training=training)
        b1 = self.conv_b1_2(b1, training=training)
        b1 = self.conv_b1_3(b1, training=training)
        b1 = self.conv_b1_4(b1, training=training)

        b2 = self.maxpool_b2_1(inputs)

        output = tf.keras.layers.concatenate([b0, b1, b2], axis=-1)
        return output


class InceptionModule5(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionModule5, self).__init__()
        self.conv1 = BasicConv2D(filters=320,
                                 kernel_size=(1, 1),
                                 strides=1,
                                 padding="same")
        self.conv2 = BasicConv2D(filters=384,
                                 kernel_size=(1, 1),
                                 strides=1,
                                 padding="same")
        self.conv3 = BasicConv2D(filters=448,
                                 kernel_size=(1, 1),
                                 strides=1,
                                 padding="same")
        self.conv4 = BasicConv2D(filters=384,
                                 kernel_size=(1, 3),
                                 strides=1,
                                 padding="same")
        self.conv5 = BasicConv2D(filters=384,
                                 kernel_size=(3, 1),
                                 strides=1,
                                 padding="same")
        self.conv6 = BasicConv2D(filters=384,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding="same")
        self.conv7 = BasicConv2D(filters=192,
                                 kernel_size=(1, 1),
                                 strides=1,
                                 padding="same")
        self.avgpool = tf.keras.layers.AvgPool2D(pool_size=(3, 3),
                                                 strides=1,
                                                 padding="same")

    def call(self, inputs, training=None, **kwargs):
        b0 = self.conv1(inputs, training=training)

        b1 = self.conv2(inputs, training=training)
        b1_part_a = self.conv4(b1, training=training)
        b1_part_b = self.conv5(b1, training=training)
        b1 = tf.keras.layers.concatenate([b1_part_a, b1_part_b], axis=-1)

        b2 = self.conv3(inputs, training=training)
        b2 = self.conv6(b2, training=training)
        b2_part_a = self.conv4(b2, training=training)
        b2_part_b = self.conv5(b2, training=training)
        b2 = tf.keras.layers.concatenate([b2_part_a, b2_part_b], axis=-1)
        b3 = self.avgpool(inputs)
        b3 = self.conv7(b3, training=training)

        output = tf.keras.layers.concatenate([b0, b1, b2, b3], axis=-1)
        return output


class InceptionAux(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        super(InceptionAux, self).__init__()
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=(5, 5),
                                                  strides=3,
                                                  padding="same")
        self.conv1 = BasicConv2D(filters=128,
                                 kernel_size=(1, 1),
                                 strides=1,
                                 padding="same")
        self.conv2 = BasicConv2D(filters=768,
                                 kernel_size=(5, 5),
                                 strides=1,
                                 padding="same")
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.flat = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.linear)

    def call(self, inputs, training=None, **kwargs):
        output = self.avg_pool(inputs)
        output = self.conv1(output, training=training)
        output = self.conv2(output, training=training)
        output = self.global_avg_pool(output)
        output = self.flat(output)
        output = self.fc(output)

        return output