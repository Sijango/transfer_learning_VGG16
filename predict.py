from lib_image_search import get_data, config
import tensorflow as tf

from lib_image_search.loss_and_metrics import custom_loss, mean_iou

tf.config.run_functions_eagerly(True)

if __name__ == '__main__':
    x_train, y_train = get_data.get_all_data(config.TRAIN_PATH)
    x_valid, y_valid = get_data.get_all_data(config.VALID_PATH)

    model = tf.keras.models.load_model(config.MODEL_PATH, custom_objects={'custom_loss': custom_loss, 'mean_iou': mean_iou})

    loss, class_loss, out_loss, test_acc, iou = model.evaluate(x_valid, y_valid, verbose=2)

    print('\nmIoU {:.2f}%, classification accuracy {:.2f}%, {} train, {} valid.'.format(iou * 100, test_acc * 100,
                                                                                        len(x_train),
                                                                                        len(x_valid)))
