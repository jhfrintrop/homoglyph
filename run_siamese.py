# -*- coding: utf-8 -*-


import logging
import os
import random
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from time import time

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, Lambda, Flatten, Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras import backend as K
from keras.optimizers import RMSprop
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_curve, auc, roc_auc_score


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)-8s] %(processName)s:%(process)-5d %(name)s: %(message)s'
)

TEST = True
FAST = True  # If True, then it runs on a very small dataset (and results won't be that great)

DATASET_TYPE = 'domain'
# DATASET_TYPE = 'process'

OUTPUT_DIR = 'output'


def _generate_img(string, font_location, font_size, image_size, text_location):
    font = ImageFont.truetype(font_location, font_size)
    img = Image.new('F', image_size)
    dimg = ImageDraw.Draw(img)
    dimg.text(text_location, string.lower(), font=font)

    img = np.expand_dims(img, axis=0)

    return img


def _euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y) + np.random.rand() * 0.0001, axis=1, keepdims=True))


def _eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def _contrastive_loss(y_true, y_pred):
    """Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)), axis=-1, keepdims=False)


def _build_model(data_shape):
    model = Sequential()

    model.add(Convolution2D(128, 5, 5, input_shape=data_shape))
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(32))

    input_a = Input(shape=data_shape)
    input_b = Input(shape=data_shape)

    processed_a = model(input_a)
    processed_b = model(input_b)

    distance = Lambda(_euclidean_distance, output_shape=_eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(input=[input_a, input_b], output=distance)

    # train
    rms = RMSprop()
    model.compile(loss=_contrastive_loss, optimizer=rms)

    return model


def _save_neural_network(model, output_dir, dataset_type):
    logger.info('Saving neural network to %s', output_dir)
    t0 = time()

    if not os.path.isdir(output_dir):
        logger.debug('Creating %d', output_dir)
        os.mkdir(output_dir)

    logger.debug('Saving model as h5')
    model.save(os.path.join(output_dir, f'{dataset_type}_model.h5'), overwrite=True)

    logger.debug('Saving model as json')
    with open(os.path.join(output_dir, f'{dataset_type}_model.json'), 'w') as f:
        f.write(model.to_json())

    logger.debug('Saving weights as h5')
    model.save_weights(os.path.join(output_dir, f'{dataset_type}_weights.h5'), overwrite=True)

    t1 = time()
    logger.info('Saved neural network to %s, took %fs', output_dir, t1 - t0)


def _print_figure(results, filename: str):
    logger.info('Printing figure to %s', filename)
    t0 = time()

    fig = plt.figure()
    plt.plot(results['fpr'], results['tpr'], 'b', label=f'Siamese CNN (AUC={results["auc"]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k', lw=3, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{DATASET_TYPE.capitalize()} Spoofing - Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    fig.savefig(filename)

    t1 = time()
    logger.info('Printed figure to %s, took %fs', filename, t1 - t0)


def _create_neural_network():
    font_location = 'Arial.ttf'
    font_size = 10
    image_size = (150, 12)
    text_location = (0, 0)
    max_epochs = 25

    logger.info('Creating neural network')
    logger.debug('Font: %s %dpt', font_location, font_size)
    logger.debug('Image size: %s', image_size)
    logger.debug('Text location: %s', text_location)
    logger.debug('Max. epochs: %d', max_epochs)

    logger.info('Loading data')
    t0 = time()
    with open(os.path.join('data', f'{DATASET_TYPE}.pkl'), 'rb') as f:
        data = pickle.load(f)
    t1 = time()
    logger.info('Loaded data, took %fs', t1 - t0)

    if TEST:
        logger.info('TEST, reducing data')
        t0 = time()
        data['train'] = random.sample(data['train'], 200)
        data['validate'] = random.sample(data['validate'], 100)
        data['test'] = random.sample(data['test'], 100)
        max_epochs = 10
        t1 = time()
        logger.info('TEST, reduced data, took %fs', t1 - t0)
    elif FAST:
        logger.info('FAST, reducing data')
        t0 = time()
        data['train'] = random.sample(data['train'], 20000)
        data['validate'] = random.sample(data['validate'], 100)
        data['test'] = random.sample(data['test'], 1000)
        max_epochs = 10
        t1 = time()
        logger.info('FAST, reduced data, took %fs', t1 - t0)

    generate_img = partial(_generate_img, font_location=font_location, font_size=font_size, image_size=image_size, text_location=text_location)

    # organize data and translate from th to tf image ordering via .transpose((0, 2, 3, 1))
    logger.info('Building training data')
    t0 = time()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        X1_train = np.array(list(ex.map(generate_img, [x[0] for x in data['train']], chunksize=len(data['train']) // ex._max_workers)), dtype=np.float32).transpose((0, 2, 3, 1))
        X2_train = np.array(list(ex.map(generate_img, [x[1] for x in data['train']], chunksize=len(data['train']) // ex._max_workers)), dtype=np.float32).transpose((0, 2, 3, 1))
        y_train = [
            x[2]
            for x in data['train']
        ]
    del data['train']
    t1 = time()
    logger.info('Built training data, took %fs', t1 - t0)

    logger.info('Building validation data')
    t0 = time()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        X1_valid = np.array(list(ex.map(generate_img, [x[0] for x in data['validate']], chunksize=len(data['validate']) // ex._max_workers)), dtype=np.float32).transpose((0, 2, 3, 1))
        X2_valid = np.array(list(ex.map(generate_img, [x[1] for x in data['validate']], chunksize=len(data['validate']) // ex._max_workers)), dtype=np.float32).transpose((0, 2, 3, 1))
        y_valid = [
            x[2]
            for x in data['validate']
        ]
    del data['validate']
    t1 = time()
    logger.info('Built validation data, took %fs', t1 - t0)

    model = _build_model(data_shape=(12, 150, 1))

    logger.info('Testing how many epochs are needed')
    t0 = time()
    max_auc = 0
    epochs = 0
    for i in range(max_epochs):
        model.fit([X1_train, X2_train], y_train, batch_size=8, epochs=1)
        scores = [
            -x[0]
            for x in model.predict([X1_valid, X2_valid], verbose=1)
        ]

        curr_auc = roc_auc_score(y_valid, scores)
        if curr_auc > max_auc:
            max_auc = curr_auc
            epochs = i + 1
            logger.info('Updated best AUC from %f to %f in epoch %d', max_auc, curr_auc, epochs)
    t1 = time()
    logger.info('Tested how many epochs are needed: %d, took %fs', epochs, t1 - t0)

    logger.debug('Deleting validating data')
    t0 = time()
    del X1_valid
    del X2_valid
    t1 = time()
    logger.debug('Deleted validating data, took %fs', epochs, t1 - t0)

    logger.info('Training model with %d epochs', epochs)
    t0 = time()
    model = _build_model(data_shape=(12, 150, 1))
    model.fit([X1_train, X2_train], y_train, batch_size=8, epochs=epochs)
    t1 = time()
    logger.info('Trained model with %d epochs, took %fs', epochs, t1 - t0)

    logger.debug('Deleting training data')
    t0 = time()
    del X1_train
    del X2_train
    t1 = time()
    logger.debug('Deleted training data, took %fs', t1 - t0)

    _save_neural_network(model=model, output_dir=OUTPUT_DIR, dataset_type=DATASET_TYPE)

    logger.info('Building test data')
    t0 = time()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        X1_test = np.array(list(ex.map(generate_img, [x[0] for x in data['test']], chunksize=len(data['test']) // ex._max_workers)), dtype=np.float32).transpose((0, 2, 3, 1))
        X2_test = np.array(list(ex.map(generate_img, [x[1] for x in data['test']], chunksize=len(data['test']) // ex._max_workers)), dtype=np.float32).transpose((0, 2, 3, 1))
        y_test = [
            x[2]
            for x in data['test']
        ]
    del data['test']
    t1 = time()
    logger.info('Built test data, took %fs', t1 - t0)

    scores = [
        -x[0]
        for x in model.predict([X1_test, X2_test], verbose=1)
    ]

    logger.debug('Deleting test data')
    t0 = time()
    del X1_test
    del X2_test
    t1 = time()
    logger.debug('Deleted test data, took %fs', t1 - t0)

    logger.debug('Deleting data')
    t0 = time()
    del data
    t1 = time()
    logger.debug('Deleted data, took %fs', t1 - t0)

    fpr_siamese, tpr_siamese, _ = roc_curve(y_test, scores)
    roc_auc_siamese = auc(fpr_siamese, tpr_siamese)

    results = {
        'fpr': fpr_siamese,
        'tpr': tpr_siamese,
        'auc': roc_auc_siamese,
    }

    figure_filename = os.path.join(OUTPUT_DIR, f'{DATASET_TYPE}_roc_curve.png')
    _print_figure(results=results, filename=figure_filename)


def main():
    _create_neural_network()


if __name__ == '__main__':
    main()
