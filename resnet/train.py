import tensorflow as tf
from model import resnet
import config
from dataset import build_dataset
import math
import datetime


def get_model():
    if config.model == 'resnet18':
        model = resnet.resnet_18(config.NUM_CLASSES)
    elif config.model == 'resnet34':
        model = resnet.resnet_34(config.NUM_CLASSES)

    elif config.model == 'resnet50':
        model = resnet.resnet_50(config.NUM_CLASSES)

    elif config.model == 'resnet101':
        model = resnet.resnet_101(config.NUM_CLASSES)

    else:
        model = resnet.resnet_152(config.NUM_CLASSES)

    return model


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

model = get_model()
train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = build_dataset()

# define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizers = tf.keras.optimizers.Adadelta()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

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
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(y_true=labels, y_pred=predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizers.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def valid_step(images, labels):
    predictions = model(images, include_aux_logits=False, training=False)
    v_loss = loss_object(labels, predictions)

    valid_loss(v_loss)
    valid_accuracy(labels, predictions)


def main():
    # start training
    for epoch in range(config.EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        for step, (images, labels) in enumerate(train_dataset):
            train_step(images, labels)
            template = 'Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}'
            print(template.format(epoch + 1,
                                  config.EPOCHS,
                                  step,
                                  math.ceil(
                                      train_count / config.BATCH_SIZE),
                                  train_loss.result(),
                                  train_accuracy.result()))

        for valid_images, valid_labels in valid_dataset:
            valid_step(valid_images, valid_labels)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                  config.EPOCHS,
                                                                  train_loss.result(),
                                                                  train_accuracy.result(),
                                                                  valid_loss.result(),
                                                                  valid_accuracy.result()))

        with train_summary.as_default():
            tf.summary.scalar('train loss', train_loss.result(), step=epoch)
            tf.summary.scalar('train accuracy', train_accuracy.result(), step=epoch)
        with test_summary.as_default():
            tf.summary.scalar('valid loss', valid_loss.result(), step=epoch)
            tf.summary.scalar('valid accuracy', valid_accuracy.result(), step=epoch)

        manager.save()


if __name__ == '__main__':
    main()
    print(1)
