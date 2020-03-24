import tensorflow as tf
from inception_module import InceptionModule1, Preprocess, InceptionModule3, InceptionModule2, InceptionModule4, \
    InceptionModule5, InceptionAux, BasicConv2D
from collections import namedtuple

InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])


class InceptionV3(tf.keras.Model):
    def __init__(self, num_class, aux_logits=True):
        super(InceptionV3, self).__init__()

        self.aux_logits = aux_logits
        self.preprocess = Preprocess()
        self.block1 = tf.keras.Sequential([
            InceptionModule1(32),
            InceptionModule1(64),
            InceptionModule1(64),
        ])

        self.block2 = tf.keras.Sequential([
            InceptionModule2(),
            InceptionModule3(128),
            InceptionModule3(160),
            InceptionModule3(160),
            InceptionModule3(192),
        ])

        if self.aux_logits:
            self.AuxLogits = InceptionAux(num_classes=num_class)

        self.block3 = tf.keras.Sequential([
            InceptionModule4(),
            InceptionModule5(),
            InceptionModule5(),
        ])

        self.maxpool = tf.keras.layers.MaxPool2D((8, 8), strides=1, padding='same')
        self.dropout = tf.keras.layers.Dropout(rate=0.4)
        self.conv2d = BasicConv2D(filters=1000,
                                  kernel_size=(1, 1),
                                  strides=1,
                                  padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(units=num_class, activation=tf.keras.activations.relu)

    def call(self, inputs, training=None, mask=None, include_aux_logits=True):
        x = self.preprocess(inputs, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        if include_aux_logits and self.aux_logits:
            aux = self.AuxLogits(x)
        x = self.block_3(x, training=training)
        x = self.avg_pool(x)
        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        if include_aux_logits and self.aux_logits:
            return InceptionOutputs(x, aux)
        return x
