import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets
import os
import datetime


def processing_data(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)

    return x, y


def mnist_dataset():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    # Create Dataset
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Pre-processing data
    ds_train = ds_train.map(processing_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(processing_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds_train = ds_train.shuffle(60000).batch(100)
    ds_test = ds_test.batch(100)

    return ds_train, ds_test


model = tf.keras.Sequential([
    layers.Reshape(target_shape=(28, 28, 1), input_shape=(28, 28,)),
    layers.Conv2D(2, 3, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
    layers.Conv2D(4, 3, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
    layers.Flatten(),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dropout(rate=0.4),
    layers.Dense(10)
])

optimizers = tf.keras.optimizers.Adam()

# Create checkpoint
checkpoint_dir = 'checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

checkpoint = tf.train.Checkpoint(model=model, optimizers=optimizers)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=2)

# Load lastest checkpoint
status = checkpoint.restore(manager.latest_checkpoint)

# Create summary log file
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


@tf.function
def compute_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))


@tf.function
def compute_accuracy(logits, labels):
    pred = tf.argmax(logits, axis=1)

    return tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))


@tf.function
def train_step(features, labels):
    with tf.GradientTape() as tape:
        logits = model(features)
        loss = compute_loss(logits, labels)

    # Compute gradient
    grads = tape.gradient(loss, model.trainable_variables)

    # Update Weight
    optimizers.apply_gradients(zip(grads, model.trainable_variables))

    # Compute accuracy
    accuracy = compute_accuracy(logits, labels)

    return loss, accuracy


def test(dataset):
    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    avg_accuracy = tf.keras.metrics.Mean('loss', dtype=tf.float32)

    for (x, y) in dataset:
        logits = model(x)
        avg_loss(compute_loss(logits, y))
        avg_accuracy(compute_accuracy(logits, y))

    return avg_loss.result(), avg_accuracy.result()


def train(epochs, train_ds, test_ds):
    train_loss, train_accuracy = 0.0, 0.0
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_ds):
            train_loss, train_accuracy = train_step(x, y)
        # test
        test_loss, test_accuracy = test(test_ds)

        # save checkpoint
        manager.save()

        # write summary
        with train_summary_writer.as_default():
            tf.summary.scalar('train loss', train_loss, step=epoch)
            tf.summary.scalar('train accuracy', train_accuracy, step=epoch)
        with test_summary_writer.as_default():
            tf.summary.scalar('test loss', test_loss, step=epoch)
            tf.summary.scalar('test accuracy', test_accuracy, step=epoch)

        template = 'Epoch {}, Loss: {:.3f}, Accuracy: {}%, Test Loss: {:.3f}, Test Accuracy: {}%'
        print(template.format(epoch + 1,
                              train_loss,
                              train_accuracy * 100,
                              test_loss,
                              test_accuracy * 100))


train_ds, test_ds = mnist_dataset()
train(10, train_ds, test_ds)
