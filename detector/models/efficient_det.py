import tensorflow as tf
from models.resnet import resnet_50
import configs


def get_model(cf):
    # TODO: build your model
    model = resnet_50()
    model.build(input_shape=(None, configs.INPUT_SHAPE[0], configs.INPUT_SHAPE[1],3))
    model.summary()
    return model


# create model
    model = get_model()

    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adadelta()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')