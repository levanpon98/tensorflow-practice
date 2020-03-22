import os
import datetime

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers


class VGG16(models.Model):

    def __init__(self, input_shape):
        """

        :param input_shape: [32, 32, 3]
        """
        super(VGG16, self).__init__()

        weight_decay = 0.000
        self.num_classes = 10

        model = models.Sequential()

        model.add(layers.Conv2D(64, (3, 3), padding='same',
                                input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes))
        # model.add(layers.Activation('softmax'))

        self.model = model

    def call(self, x):
        x = self.model(x)

        return x


def preprocessing_data(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int32)

    mean = tf.math.reduce_mean(x)
    std = tf.math.reduce_std(x)
    x = (x - mean) / (std + 1e-7)

    return x, y


def build_dataset():
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    ds_train = ds_train.map(preprocessing_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(preprocessing_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds_train = ds_train.shuffle(50000).batch(100)
    ds_test = ds_test.batch(100)

    return ds_train, ds_test


model = VGG16([32, 32, 3])
optimizers = tf.keras.optimizers.Adam()

train_accuracy = tf.keras.metrics.CategoricalCrossentropy(name='train_accuracy')
test_accuracy = tf.keras.metrics.CategoricalCrossentropy(name='test_accuracy')

# Create checkpoint
checkpoint_dir = 'checkpoints/'
checkpoint = tf.train.Checkpoint(model=model, optimizers=optimizers)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=1)
status = checkpoint.restore(manager.latest_checkpoint)

current_time = datetime.datetime.now().strftime('%Y%m%d-H%M%S')
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary = tf.summary.create_file_writer(train_log_dir)
test_summary = tf.summary.create_file_writer(test_log_dir)


@tf.function
def compute_loss(logits, labels):
    return tf.math.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    )


@tf.function
def train_step(features, labels):
    labels = tf.squeeze(labels, axis=1)

    labels = tf.one_hot(labels, depth=10)

    with tf.GradientTape() as tape:
        logits = model(features)
        loss = compute_loss(logits, labels)
    # Compute Gradient
    grads = tape.gradient(loss, model.trainable_variables)
    grads = [tf.clip_by_norm(g, 1.5) for g in grads]
    # Update weight
    optimizers.apply_gradients(zip(grads, model.trainable_variables))
    # Compute accuracy
    train_accuracy(logits, labels)

    return loss


def test(dataset):
    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

    for x, y in dataset:
        y = tf.squeeze(y, axis=1)
        y = tf.one_hot(y, depth=10)

        logits = model(x)
        avg_loss(compute_loss(logits, y))
        test_accuracy(logits, y)

    return avg_loss.result()


def train(epochs, train_ds, test_ds):
    train_loss = 0.0
    for epoch in range(epochs):
        train_accuracy.reset_states()
        test_accuracy.reset_states()
        for step, (x, y) in enumerate(train_ds):
            train_loss = train_step(x, y)

        test_loss = test(test_ds)

        template = "Epoch {}: train_loss: {:2f} train_accuracy: {}% test_loss: {:2f} test_accuracy: {}%"
        print(template.format(
            epoch,
            train_loss,
            train_accuracy.result() * 100,
            test_loss,
            test_accuracy.result() * 100
        ))

        with train_summary.as_default():
            tf.summary.scalar('train loss', train_loss, step=epoch)
            tf.summary.scalar('train accuracy', train_accuracy.result(), step=epoch)
        with test_summary.as_default():
            tf.summary.scalar('test loss', test_loss, step=epoch)
            tf.summary.scalar('test accuracy', test_accuracy.result(), step=epoch)

        manager.save()


train_ds, test_ds = build_dataset()
train(1, train_ds, test_ds)
