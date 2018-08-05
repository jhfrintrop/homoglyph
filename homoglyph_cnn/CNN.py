# -*- coding: utf-8 -*-


import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import partial
from tempfile import TemporaryDirectory
from time import time
from typing import Tuple, List, Dict, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, Lambda, Flatten, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, model_from_json, load_model
from keras import backend as K
from keras.optimizers import RMSprop
from sklearn.metrics import roc_curve, auc, roc_auc_score

from homoglyph_cnn.generate_img import generate_img as _generate_img
from homoglyph_cnn.Archive import Archive


logger = logging.getLogger(__name__)


class CNN(object):
    def __init__(self, font_location: str, font_size: int, image_size: Tuple[int, int], text_location: Tuple[int, int]):
        self.font_location = font_location
        self.font_size = font_size
        self.image_size = image_size
        self.text_location = text_location

        self.model: Model = None

    @staticmethod
    def _euclidean_distance(vects) -> float:
        x, y = vects

        return K.sqrt(K.sum(K.square(x - y) + np.random.rand() * 0.0001, axis=1, keepdims=True))

    @staticmethod
    def _eucl_dist_output_shape(shapes):
        shape1, shape2 = shapes

        return shape1[0], 1

    @staticmethod
    def _contrastive_loss(y_true: float, y_pred: float) -> float:
        """Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        """

        margin = 1
        x = y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0))
        return K.mean(x, axis=-1, keepdims=False)

    @classmethod
    def _build_model(cls, data_shape: Tuple[int, int, int]) -> Model:
        model = Sequential()

        # TODO: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(128, (5, 5), input_shape=(12, 150, ...)`
        # https://github.com/keras-team/keras/wiki/Keras-2.0-release-notes#convolutional-layers
        model.add(Conv2D(128, (5, 5), input_shape=data_shape))
        model.add(LeakyReLU(alpha=0.1))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        # TODO: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(64, (3, 3))`
        # https://github.com/keras-team/keras/wiki/Keras-2.0-release-notes#convolutional-layers
        model.add(Conv2D(64, (3, 3)))
        model.add(LeakyReLU(alpha=0.1))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(32))

        input_a = Input(shape=data_shape)
        input_b = Input(shape=data_shape)

        processed_a = model(input_a)
        processed_b = model(input_b)

        distance = Lambda(cls._euclidean_distance, output_shape=cls._eucl_dist_output_shape)([processed_a, processed_b])

        # TODO: UserWarning: Update your `Model` call to the Keras 2 API: `Model(inputs=[<tf.Tenso..., outputs=Tensor("la...)`
        # https://github.com/keras-team/keras/wiki/Keras-2.0-release-notes#models
        model = Model(inputs=[input_a, input_b], outputs=distance)

        rms = RMSprop()
        model.compile(loss=cls._contrastive_loss, optimizer=rms)

        return model

    def _build_data(self, data: List[Tuple[str, str, float]], max_workers: int = None) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        if max_workers is None or max_workers <= 0:
            max_workers = os.cpu_count() or 1

        generate_img = partial(_generate_img, font_location=self.font_location, font_size=self.font_size,
                               image_size=self.image_size, text_location=self.text_location)

        logger.debug('Generating images, %d worker processes', max_workers)
        t0 = time()

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            imgs_1 = ex.map(generate_img, [x[0] for x in data], chunksize=max(1, len(data) // max_workers))
            imgs_2 = ex.map(generate_img, [x[1] for x in data], chunksize=max(1, len(data) // max_workers))

        t1 = time()
        logger.debug('Generated images, took %fs', t1 - t0)

        logger.debug('Organizing data and translating from th to tf image ordering via .transpose((0, 2, 3, 1))')
        t0 = time()

        X1 = np.array(list(imgs_1), dtype=np.float32).transpose((0, 2, 3, 1))
        X2 = np.array(list(imgs_2), dtype=np.float32).transpose((0, 2, 3, 1))
        y = [
            x[2]
            for x in data
        ]

        t1 = time()
        logger.debug('Organized data and translated from th to tf image ordering via .transpose((0, 2, 3, 1)), took %fs', t1 - t0)

        return X1, X2, y

    def train(self, data: Dict[str, List[Tuple[str, str, float]]], max_epochs: int = 25, verbose: bool = False) -> Dict[str, Any]:
        data_shape = (12, 150, 1)
        batch_size = 8
        verbose = 1 if verbose or logger.getEffectiveLevel() <= logging.DEBUG else 0

        logger.debug('Training neural network')
        logger.debug('Font: %s %dpt', self.font_location, self.font_size)
        logger.debug('Image size: %s', self.image_size)
        logger.debug('Text location: %s', self.text_location)
        logger.debug('Max. epochs: %d', max_epochs)

        logger.debug('Building training data')
        t0 = time()

        X1_train, X2_train, y_train = self._build_data(data=data['train'])
        data['train'] = None
        del data['train']

        t1 = time()
        logger.debug('Built training data, took %fs', t1 - t0)

        logger.debug('Building validation data')
        t0 = time()

        X1_validate, X2_validate, y_validate = self._build_data(data=data['validate'])
        data['validate'] = None
        del data['validate']

        t1 = time()
        logger.debug('Built validation data, took %fs', t1 - t0)

        logger.debug('Creating test model')
        test_model = self._build_model(data_shape=data_shape)

        logger.debug('Testing how many epochs are needed (max. %d)', max_epochs)
        t0 = time()

        max_auc = 0
        epochs = 0
        for i in range(1, max_epochs + 1):
            logger.debug('Epoch %d, epochs %d, max. AUC %f', i, epochs, max_auc)

            logger.debug('Fitting test model')
            test_model.fit([X1_train, X2_train], y_train, batch_size=batch_size, epochs=1, verbose=verbose)

            logger.debug('Predicting scores')
            scores = [
                -x[0]
                for x in test_model.predict([X1_validate, X2_validate], verbose=verbose)
            ]

            logger.debug('Computing ROC AUC score')
            curr_auc = roc_auc_score(y_validate, scores)

            if curr_auc > max_auc:
                max_auc = curr_auc
                epochs = i
                logger.debug('Updated best AUC from %f to %f in epoch %d', max_auc, curr_auc, i)

        t1 = time()
        logger.debug('Tested how many epochs are needed: %d, took %fs', epochs, t1 - t0)

        logger.debug('Deleting validating data')
        t0 = time()

        X1_validate = None
        del X1_validate
        X2_validate = None
        del X2_validate
        y_validate = None
        del y_validate

        t1 = time()
        logger.debug('Deleted validating data, took %fs', t1 - t0)

        logger.debug('Training model with %d epochs', epochs)
        t0 = time()

        if not self.model:
            logger.debug('Creating new model')
            self.model = self._build_model(data_shape=data_shape)
            logger.debug('Fitting new model, %d epochs', epochs)
        else:
            logger.debug('Retraining loaded model, %d epochs', epochs)
        self.model.fit([X1_train, X2_train], y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)

        t1 = time()
        logger.debug('Trained model with %d epochs, took %fs', epochs, t1 - t0)

        logger.debug('Deleting training data')
        t0 = time()

        X1_train = None
        del X1_train
        X2_train = None
        del X2_train
        y_train = None
        del y_train

        t1 = time()
        logger.debug('Deleted training data, took %fs', t1 - t0)

        # self.save(output_dir=OUTPUT_DIR)

        logger.debug('Building test data')
        t0 = time()

        X1_test, X2_test, y_test = self._build_data(data=data['test'])
        data['test'] = None
        del data['test']

        t1 = time()
        logger.debug('Built test data, took %fs', t1 - t0)

        logger.debug('Predicting test data')
        t0 = time()

        scores = [
            -x[0]
            for x in self.model.predict([X1_test, X2_test], verbose=verbose)
        ]

        t1 = time()
        logger.debug('Predicted test data, took %fs', t1 - t0)

        logger.debug('Deleting test data')
        t0 = time()

        X1_test = None
        del X1_test
        X2_test = None
        del X2_test

        t1 = time()
        logger.debug('Deleted test data, took %fs', t1 - t0)

        logger.debug('Deleting data')
        t0 = time()

        data = None
        del data

        t1 = time()
        logger.debug('Deleted data, took %fs', t1 - t0)

        fpr_siamese, tpr_siamese, _ = roc_curve(y_test, scores)
        roc_auc_siamese = auc(fpr_siamese, tpr_siamese)

        scores = None
        del scores

        results = {
            'fpr': fpr_siamese,
            'tpr': tpr_siamese,
            'auc': roc_auc_siamese,
        }

        return results

    def predict(self, data: List[Tuple[str, str]], verbose: bool = False) -> List[List[float]]:
        if not self.model:
            raise RuntimeError('Model is not set. You must train or load a model to predict.')

        verbose = 1 if verbose or logger.getEffectiveLevel() <= logging.DEBUG else 0

        X1, X2, _ = self._build_data(data=[
            i + (0,)
            for i in data
        ])

        prediction = self.model.predict([X1, X2], verbose=verbose)

        return prediction

    def save(self, filename: str, version: int = None):
        if version is None:
            version = Archive._default_version

        logger.debug('Saving to archive %s, version %d', filename, version)
        t0 = time()

        m = {
            1: self._save_v1,
            2: self._save_v2,
        }.get(version, None)

        if not m:
            raise NotImplementedError(f'Unknown version {version:d}')

        m(filename)

        t1 = time()
        logger.debug('Saved to archive %s, took %fs', filename, t1 - t0)

    def _save_v1(self, filename: str):
        with TemporaryDirectory() as temp_dir:
            model_json_file = os.path.join(temp_dir, 'model.json')
            weights_h5_file = os.path.join(temp_dir, 'weights.h5')

            logger.debug('Saving model as json to %s', model_json_file)
            with open(model_json_file, 'w') as f:
                f.write(self.model.to_json())
                f.flush()

            logger.debug('Saving weights as h5 to %s', weights_h5_file)
            self.model.save_weights(weights_h5_file, overwrite=True)

            logger.debug('Writing archive %s', filename)
            archive = Archive(filename=filename)
            archive.write(
                version=1,
                font_file=self.font_location,
                model_file=model_json_file,
                weights_file=weights_h5_file,
                font_size=self.font_size,
                image_size=self.image_size,
                text_location=self.text_location
            )

    def _save_v2(self, filename: str):
        with TemporaryDirectory() as temp_dir:
            model_h5_file = os.path.join(temp_dir, 'model.h5')

            logger.debug('Saving model as h5 to %s', model_h5_file)
            self.model.save(model_h5_file, overwrite=True)

            logger.debug('Writing archive %s', filename)
            archive = Archive(filename=filename)
            archive.write(
                version=2,
                font_filename=self.font_location,
                model_filename=model_h5_file,
                font_size=self.font_size,
                image_size=self.image_size,
                text_location=self.text_location
            )

    @classmethod
    @contextmanager
    def load(cls, filename: str) -> 'CNN':
        logger.debug('Loading from archive %s', filename)
        t0 = time()

        archive = Archive(filename=filename)
        version = archive.version

        m = {
            1: cls._load_v1,
            2: cls._load_v2,
        }.get(version, None)

        if not m:
            raise NotImplementedError(f'Unknown version {version:d}')

        with m(archive) as cnn:
            t1 = time()
            logger.debug('Loaded from archive %s, took %fs', filename, t1 - t0)

            yield cnn

    @classmethod
    @contextmanager
    def _load_v1(cls, archive: Archive) -> 'CNN':
        _font_file, model_file, _weights_file, font_size, image_size, text_location = archive.read()

        with TemporaryDirectory() as temp_dir:
            font_file_name = os.path.join(temp_dir, _font_file.name)
            weights_file_name = os.path.join(temp_dir, _weights_file.name)

            logger.debug('Copying font to %s', font_file_name)
            with open(font_file_name, 'wb') as f:
                shutil.copyfileobj(fsrc=_font_file, fdst=f)

            logger.debug('Instantiating CNN')
            cnn = cls(font_location=font_file_name, font_size=font_size, image_size=image_size,
                      text_location=text_location)

            logger.debug('Loading model from json')
            cnn.model = model_from_json(model_file.read())

            logger.debug('Copying weights file to %s', weights_file_name)
            with open(weights_file_name, 'wb') as f:
                shutil.copyfileobj(fsrc=_weights_file, fdst=f)

            logger.debug('Loading weights from %s', weights_file_name)
            cnn.model.load_weights(weights_file_name)

            yield cnn

    @classmethod
    @contextmanager
    def _load_v2(cls, archive: Archive) -> 'CNN':
        _font_file, _model_file, font_size, image_size, text_location = archive.read()

        with TemporaryDirectory() as temp_dir:
            font_file_name = os.path.join(temp_dir, _font_file.name)
            model_file_name = os.path.join(temp_dir, _model_file.name)

            logger.debug('Copying font to %s', font_file_name)
            with open(font_file_name, 'wb') as f:
                shutil.copyfileobj(fsrc=_font_file, fdst=f)

            logger.debug('Instantiating CNN')
            cnn = cls(font_location=font_file_name, font_size=font_size, image_size=image_size,
                      text_location=text_location)

            logger.debug('Copying model to %s', model_file_name)
            with open(model_file_name, 'wb') as f:
                shutil.copyfileobj(fsrc=_model_file, fdst=f)

            logger.debug('Loading model from %s', model_file_name)
            cnn.model = load_model(model_file_name, custom_objects={'_contrastive_loss': cls._contrastive_loss})

            yield cnn

    @staticmethod
    def print_figure(results: Dict[str, Any], filename: str, dataset_type: str):
        logger.debug('Printing figure to %s', filename)
        t0 = time()

        fig = plt.figure()
        plt.plot(results['fpr'], results['tpr'], 'b', label=f'Siamese CNN (AUC={results["auc"]:0.2f})')
        plt.plot([0, 1], [0, 1], 'k', lw=3, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{dataset_type.capitalize()} Spoofing - Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        fig.savefig(filename)

        t1 = time()
        logger.debug('Printed figure to %s, took %fs', filename, t1 - t0)
