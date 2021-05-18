from tensorflow import losses

from lib_image_search import config, get_data
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# import matplotlib.pyplot as plt
import tensorflow as tf

from lib_image_search.loss_and_metrics import custom_loss, mean_iou

tf.config.run_functions_eagerly(True)


if __name__ == '__main__':
    x_train, y_train = get_data.get_all_data(config.TRAIN_PATH)
    x_valid, y_valid = get_data.get_all_data(config.VALID_PATH)

    vgg = VGG16(weights="imagenet", include_top=False,
                input_tensor=Input(shape=(220, 220, 3)))

    vgg.trainable = False

    flatten = vgg.output
    flatten = Flatten()(flatten)

    bboxHead = Dense(128, activation="relu")(flatten)
    bboxHead = Dense(64, activation="relu")(bboxHead)
    bboxHead = Dense(32, activation="relu")(bboxHead)
    bboxHead_out = Dense(4, activation="sigmoid", name='out')(bboxHead)

    class_Head = Dense(32, activation="relu")(flatten)
    class_Head_out = Dense(1, activation="sigmoid", name='class')(class_Head)

    model = Model(inputs=vgg.inputs, outputs=[class_Head_out, bboxHead_out])

    model.compile(optimizer=Adam(lr=config.INIT_LR),
                  loss={
                      'class': losses.BinaryCrossentropy(),
                      'out': custom_loss
                  },
                  metrics={
                      'class': 'accuracy',
                      'out': mean_iou
                  })

    print(model.summary())

    H = model.fit(
        x_train, y_train,
        validation_data=(x_train, y_train),
        batch_size=config.BATCH_SIZE,
        epochs=config.NUM_EPOCHS,
        verbose=1)

    model.save(config.MODEL_PATH, save_format="h5")

    # N = config.NUM_EPOCHS
    # plt.style.use("ggplot")
    # plt.figure()
    # plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    # plt.title("Bounding Box Regression Loss on Training Set")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Loss")
    # plt.legend(loc="lower left")
    # plt.savefig(config.PLOT_PATH)