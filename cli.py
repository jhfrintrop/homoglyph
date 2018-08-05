#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
import re
from typing import Tuple

import click


class LoglevelParamType(click.ParamType):
    name = 'loglevel'

    def convert(self, value: str, param: str, ctx: click.Context) -> int:
        numeric_level = getattr(logging, value.upper(), None)

        if not isinstance(numeric_level, int):
            self.fail(f'Invalid log level {value}')

        return numeric_level


class LogFormatParamType(click.ParamType):
    name = 'log-format'


class ThresholdParamType(click.ParamType):
    name = 'threshold'

    def convert(self, value: str, param: str, ctx: click.Context) -> float:
        try:
            threshold = float(value)
        except ValueError as e:
            raise click.BadParameter(f'Threshold must be a float greater than 0, was "{value}"')

        if threshold <= 0:
            raise click.BadParameter(f'Threshold must be greater than 0, was {value}')

        return threshold


LOGLEVEL = LoglevelParamType()
LOG_FORMAT = LogFormatParamType()
THRESHOLD = ThresholdParamType()


@click.group(name='cnn', help='Work with a siamese convolutional neural network')
@click.option('--loglevel', type=LOGLEVEL, default='warning', show_default=True, help='The minimal loglevel to log.')
@click.option('--log-format', type=LOG_FORMAT, default='[%(asctime)s] [%(levelname)-8s] %(processName)s:%(process)d %(name)s: %(message)s', show_default=True, help='The log format or the name of the log format.')
def cnn_group(loglevel: int, log_format: str):
    logging.basicConfig(level=loglevel, format=log_format)


@cnn_group.command(name='train', help='Train or retrain a neural network')
@click.argument('font', type=click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True), required=True)
@click.argument('data', type=click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True), required=True)
@click.argument('archive', type=click.Path(exists=False, writable=True, dir_okay=False, resolve_path=True), required=False, default='model.zip')
@click.option('--font-size', type=click.IntRange(min=5, max=None), default=10, show_default=True, help='The font size')
@click.option('--image-size', type=click.Tuple([click.INT, click.INT]), default=(150, 12), show_default=True, help='The size of the image')
@click.option('--text-location', type=click.Tuple([click.INT, click.INT]), default=(0, 0), show_default=True, help='The starting location of the text')
@click.option('--max-epochs', type=click.IntRange(min=1, max=None), default=25, show_default=True, help='The maximum number of epochs to train')
@click.option('--verbose/--quiet', 'verbose', is_flag=True, default=False, show_default=True, help='If the progress of the training steps should be shown')
@click.option('--model', type=click.Path(exists=True, readable=True, resolve_path=True), default=None, show_default=True, help='The path of the archive of the model which should be retrained')
@click.option('--test', is_flag=True, default=False, show_default=True, help='If this is a test run with very limited dataset size')
@click.option('--fast', is_flag=True, default=False, show_default=True, help='If this is a fast run with limited dataset size')
@click.option('--version', type=click.IntRange(min=1, max=None), default=None, show_default=True, help='Which archive version to use')
def cnn_train(font: str, data: str, archive: str, font_size: int, image_size: Tuple[int, int], text_location: Tuple[int, int], max_epochs: int, verbose: bool, model: str, test: bool, fast: bool, version: int):
    import pickle
    import random

    from homoglyph_cnn.CNN import CNN
    from homoglyph_cnn.Archive import Archive

    with open(data, 'rb') as f:
        data = pickle.load(f)

    if test:
        click.echo('TEST, reducing data and max. epochs')
        data['train'] = random.sample(data['train'], 200)
        data['validate'] = random.sample(data['validate'], 100)
        data['test'] = random.sample(data['test'], 100)
        max_epochs = 5
    elif fast:
        click.echo('FAST, reducing data and max. epochs')
        data['train'] = random.sample(data['train'], 20000)
        data['validate'] = random.sample(data['validate'], 100)
        data['test'] = random.sample(data['test'], 1000)
        max_epochs = 10

    if model:
        a = Archive(filename=model)
        if a.version != 2:
            raise RuntimeError(f'A model of version {a.version} does not support retraining')

        click.echo(f'Retraining model from {model}')
        with CNN.load(filename=model) as cnn:
            cnn.train(data=data, max_epochs=max_epochs, verbose=verbose)
            cnn.save(filename=archive if archive else model, version=a.version)
    else:
        cnn = CNN(font_location=font, font_size=font_size, image_size=image_size, text_location=text_location)
        cnn.train(data=data, max_epochs=max_epochs, verbose=verbose)
        cnn.save(filename=archive, version=version)


@cnn_group.command(name='predict', help='Predict the similarity of pairs of strings')
@click.argument('model', type=click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True), required=True)
@click.argument('domains', type=click.STRING, required=False, nargs=-1)
@click.option('--threshold', type=THRESHOLD, default=None, show_default=True, help='If given display only results where the prediction is lower or equal this threshold')
def cnn_predict(model: str, domains: Tuple[str, ...], threshold: float):
    from homoglyph_cnn.CNN import CNN

    if not domains:
        domains = [
            line.strip()
            for line in click.get_text_stream('stdin').readlines()
        ]

    regex = re.compile('[ ,;]')
    domains = [
        tuple(map(lambda i: i.strip(), regex.split(string=line.strip(), maxsplit=1)))
        for line in domains
    ]

    with CNN.load(filename=model) as cnn:
        prediction = cnn.predict(data=domains)

        results = zip(domains, prediction.tolist())

        if threshold is not None and threshold > 0:
            def func(i):
                _, [p] = i

                return p <= threshold

            results = filter(func, results)

        for (d1, d2), [p] in results:
            click.echo(f'{d1} ~ {d2} = {p}')


@cnn_group.group(name='archive', help='Manage archives')
def cnn_archive_group():
    pass


@cnn_archive_group.command(name='pack', help='Create an v2 archive from the given files')
@click.argument('font', type=click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True), required=True)
@click.argument('model', type=click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True), required=True)
@click.argument('archive', type=click.Path(exists=False, writable=True, dir_okay=False, resolve_path=True), required=False, default='model.zip')
@click.option('--font-size', type=click.IntRange(min=5, max=None), default=10, show_default=True, help='The font size')
@click.option('--image-size', type=click.Tuple([click.INT, click.INT]), default=(150, 12), show_default=True, help='The size of the image')
@click.option('--text-location', type=click.Tuple([click.INT, click.INT]), default=(0, 0), show_default=True, help='The starting location of the text')
def cnn_archive_pack(archive: str, font: str, model: str, font_size: int, image_size: Tuple[int, int], text_location: Tuple[int, int]):
    from homoglyph_cnn.Archive import Archive

    archive = Archive(filename=archive)

    archive.write(
        version=2,
        font_filename=font,
        model_filename=model,
        font_size=font_size,
        image_size=image_size,
        text_location=text_location
    )


@cnn_archive_group.command(name='unpack', help='Extract the given archive')
@click.argument('archive', type=click.Path(exists=True, readable=True, resolve_path=True), required=True)
@click.argument('output', type=click.Path(exists=True, writable=True, file_okay=False, resolve_path=True), required=False, default=None)
def cnn_archive_unpack(archive: str, output: str):
    from zipfile import ZipFile

    with ZipFile(file=archive, mode='r') as f:
        f.extractall(path=output)


if __name__ == '__main__':
    cnn_group = click.help_option()(cnn_group)
    cnn_group = click.version_option()(cnn_group)
    cnn_group(auto_envvar_prefix='CNN')
