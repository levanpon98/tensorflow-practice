import os
import tensorflow as tf
import progressbar

AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocessing_mnist(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y


def mnist_dataset():
    """
    Create MNIST dataset

    :return: tf.data.Dataset
    """
    # Load data set
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Create a `Dataset` whose elements are slices of the given tensors
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # But we definitely convert the feature and the labels to a Dataset
    # and then combined after

    # x = tf.data.Dataset.from_tensor_slices(x)
    # y = tf.data.Dataset.from_tensor_slice(y)
    # ds = tf.data.Dataset.zip((x, y))

    # Maps map_func across the elements of this dataset.
    ds_train = ds_train.map(preprocessing_mnist)
    ds_test = ds_test.map(preprocessing_mnist)

    # You can set the num_parallel_calls parameter to enable parallel process
    # with available CPU
    # ds = ds.map(preprocessing_mnist, num_parallel_calls=AUTOTUNE)

    ds_train = ds_train.shuffle(60000).batch(100)
    ds_test = ds_test.shuffle(60000).batch(100)
    return ds_train, ds_test


# Define a custom model
model = tf.keras.Sequential([
    tf.keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(10)
])

# Define optimizers function
optimizer = tf.keras.optimizers.Adam()


@tf.function
def compute_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))


@tf.function
def compute_accuracy(logits, labels):
    pred = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))


@tf.function
def train_step(x_train, y_train):
    with tf.GradientTape() as tape:
        logist = model(x_train)
        loss = compute_loss(logist, y_train)

    # Compute Gradient
    grads = tape.gradient(loss, model.trainable_variables)

    # Update to Weights
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Compute accuracy
    accuracy = compute_accuracy(logist, y_train)

    return loss, accuracy


def train(epochs, train_ds):
    loss = 0.0
    accuracy = 0.0

    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_ds):
            loss, accuracy = train_step(x, y)

            if step % 500 == 0:
                print('epoch {}: loss - {:.2f} accuracy - {:.2f}'.format(epoch, loss.numpy(), accuracy.numpy()))
        print('Final epoch {}: loss - {:.2f} accuracy - {:.2f}'.format(epoch, loss.numpy(), accuracy.numpy()))


train_ds, test_ds = mnist_dataset()
train(10, train_ds)
