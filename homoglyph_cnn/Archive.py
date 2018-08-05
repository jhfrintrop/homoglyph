# -*- coding: utf-8 -*-


import json
import logging
from functools import partial
from hashlib import md5
from os.path import splitext
from time import time
from typing import Tuple
from zipfile import ZipFile

from fontTools.ttLib import TTFont


FONT_SPECIFIER_NAME_ID = 4
FONT_SPECIFIER_FAMILY_ID = 1

logger = logging.getLogger(__name__)


def get_font_info(font_file):
    with TTFont(font_file) as font:
        name = None
        family = None

        for record in font['name'].names:
            if b'\x00' in record.string:
                name_str = record.string.decode('utf-16-be')
            else:
                name_str = record.string.decode('utf-8')

            if record.nameID == FONT_SPECIFIER_NAME_ID and not name:
                name = name_str
            elif record.nameID == FONT_SPECIFIER_FAMILY_ID and not family:
                family = name_str

            if name and family:
                break

        return name, family


def hash_file_by_name(filename: str) -> str:
    with open(filename, 'rb') as f:
        return hash_file(file=f)


def hash_file(file) -> str:
    m = md5()
    blocksize = min(m.block_size * 1000, 1000000)
    func = partial(file.read, blocksize)

    for chunk in iter(func, b''):
        m.update(chunk)

    return m.hexdigest()


class Archive(object):
    _default_version = 1
    _writers = {}
    _readers = {}
    _version = None

    def __init__(self, filename: str):
        super(Archive, self).__init__()

        self.filename = filename

    @property
    def version(self):
        if self._version is None:
            with ZipFile(file=self.filename, mode='r') as z:
                meta = json.loads(z.read('meta.json'))
                self._version = int(meta['version'])

        return self._version

    def write(self, version: int = None, *args, **kwargs):
        if version is None:
            version = self._default_version

        writer = self._writers.get(version, None)

        if not writer:
            raise NotImplementedError(f'No writer registered for version {version:d}')

        kwargs['archive_filename'] = self.filename
        retval = writer(*args, **kwargs)

        logger.debug('Testing archive')
        self.test()

        return retval

    def read(self, *args, **kwargs):
        logger.debug('Testing archive')
        self.test()

        with ZipFile(file=self.filename, mode='r') as z:
            logger.debug('Loading meta')
            meta = json.loads(z.read('meta.json'))
            version = int(meta['version'])
            logger.debug('Archive version %d', version)

        reader = self._readers.get(version, None)

        if not reader:
            raise NotImplementedError(f'No reader registered for version {version:d}')

        kwargs['archive_filename'] = self.filename
        return reader(*args, **kwargs)

    def test(self) -> bool:
        logger.debug('Testing %s', self.filename)
        with ZipFile(file=self.filename, mode='r') as z:
            bad_filename = z.testzip()
            if bad_filename is not None:
                raise AssertionError(f'Bad archive {bad_filename}')

        return True

    @classmethod
    def writer(cls, version: int, default: bool = False):
        def decorator(func):
            logger.debug('Adding writer for version %d', version)
            cls._writers[version] = func

            if default:
                logger.debug('Version %d is new default', version)
                cls._default_version = version

            return func

        return decorator

    @classmethod
    def reader(cls, version: int):
        def decorator(func):
            logger.debug('Adding reader for version %d', version)
            cls._readers[version] = func

            return func

        return decorator


@Archive.writer(version=1)
def write_v1(font_file, model_file, weights_file, archive_filename, font_size: int = 10, image_size: Tuple[int, int] = (150, 12), text_location: Tuple[int, int] = (0, 0)):
    # Get the extension from the font file
    _, font_extension = splitext(font_file if isinstance(font_file, str) else font_file.name)

    # Target names (names of the files in the archive)
    meta_target = 'meta.json'
    font_target = f'font{font_extension}'
    model_target = 'model.json'
    weights_target = 'weights.h5'

    def _build_meta():
        # Checksums
        font_checksum = hash_file_by_name(filename=font_file) if isinstance(font_file, str) else hash_file(file=font_file)
        model_checksum = hash_file_by_name(filename=model_file) if isinstance(model_file, str) else hash_file(file=model_file)
        weights_checksum = hash_file_by_name(filename=weights_file) if isinstance(weights_file, str) else hash_file(file=weights_file)

        font_name, font_family = get_font_info(font_file)

        # Meta information
        meta = {
            'version': 1,
            'build': int(time()),
            'font': {
                'filename': font_target,
                'checksum': font_checksum,
                'type': font_extension,
                'name': font_name,
                'family': font_family,
            },
            'model': {
                'filename': model_target,
                'checksum': model_checksum,
            },
            'weights': {
                'filename': weights_target,
                'checksum': weights_checksum,
            },
            'image': {
                'font_size': font_size,
                'image_size': image_size,
                'text_location': text_location,
            },
        }

        return meta

    # Create and fill archive
    with ZipFile(file=archive_filename, mode='w') as z:
        logger.debug('Copying font to archive')
        if isinstance(font_file, str):
            z.write(filename=font_file, arcname=font_target)
        else:
            z.writestr(zinfo_or_arcname=font_target, data=font_file.read())

        logger.debug('Copying model to archive')
        if isinstance(model_file, str):
            z.write(filename=model_file, arcname=model_target)
        else:
            z.writestr(zinfo_or_arcname=model_target, data=model_file.read())

        logger.debug('Copying weights to archive')
        if isinstance(weights_file, str):
            z.write(filename=weights_file, arcname=weights_target)
        else:
            z.writestr(zinfo_or_arcname=weights_target, data=weights_file.read())

        logger.debug('Generating meta')
        meta = _build_meta()
        logger.debug('Writing meta to archive')
        z.writestr(zinfo_or_arcname=meta_target, data=json.dumps(meta))

    logger.debug('Wrote %s, %s and %s to %s', font_file, model_file, weights_file, archive_filename)


@Archive.writer(version=2, default=True)
def write_v2(font_filename, model_filename, archive_filename, font_size: int = 10, image_size: Tuple[int, int] = (150, 12), text_location: Tuple[int, int] = (0, 0)):
    # Get the extension from the font file
    _, font_extension = splitext(font_filename)

    # Target names (names of the files in the archive)
    meta_target = 'meta.json'
    font_target = f'font{font_extension}'
    model_target = 'model.h5'

    def _build_meta():
        # Checksums
        font_checksum = hash_file_by_name(filename=font_filename)
        model_checksum = hash_file_by_name(filename=model_filename)

        font_name, font_family = get_font_info(font_filename)

        # Meta information
        meta = {
            'version': 2,
            'build': int(time()),
            'font': {
                'filename': font_target,
                'checksum': font_checksum,
                'type': font_extension,
                'name': font_name,
                'family': font_family,
            },
            'model': {
                'filename': model_target,
                'checksum': model_checksum,
            },
            'image': {
                'font_size': font_size,
                'image_size': image_size,
                'text_location': text_location,
            },
        }

        return meta

    # Create and fill archive
    with ZipFile(file=archive_filename, mode='w') as z:
        logger.debug('Copying font to archive')
        z.write(font_filename, font_target)

        logger.debug('Copying model to archive')
        z.write(model_filename, model_target)

        logger.debug('Generating meta')
        meta = _build_meta()
        logger.debug('Writing meta to archive')
        z.writestr(meta_target, json.dumps(meta))

    logger.debug('Wrote %s and %s to %s', font_filename, model_filename, archive_filename)


@Archive.reader(version=1)
def read_v1(archive_filename: str):
    with ZipFile(file=archive_filename, mode='r') as z:
        logger.debug('Loading meta')
        meta = json.loads(z.read('meta.json'))

        logger.debug('Checking checksums')
        for t in ['font', 'model', 'weights']:
            with z.open(meta[t]['filename'], 'r') as f:
                expected_checksum = meta[t]['checksum']
                actual_checksum = hash_file(file=f)

                if expected_checksum != actual_checksum:
                    raise AssertionError(f'{t} checksum differs, expected {expected_checksum}, got {actual_checksum}')
        else:
            logger.debug('All checksums ok')

        font_file = z.open(meta['font']['filename'], 'r')
        model_file = z.open(meta['model']['filename'], 'r')
        weights_file = z.open(meta['weights']['filename'], 'r')
        font_size = int(meta['image']['font_size'])
        image_size = tuple(meta['image']['image_size'])
        text_location = tuple(meta['image']['text_location'])

        return font_file, model_file, weights_file, font_size, image_size, text_location


@Archive.reader(version=2)
def read_v2(archive_filename: str):
    with ZipFile(file=archive_filename, mode='r') as z:
        logger.debug('Loading meta')
        meta = json.loads(z.read('meta.json'))

        logger.debug('Checking checksums')
        for t in ['font', 'model']:
            with z.open(meta[t]['filename'], 'r') as f:
                expected_checksum = meta[t]['checksum']
                actual_checksum = hash_file(file=f)

                if expected_checksum != actual_checksum:
                    raise AssertionError(f'{t} checksum differs, expected {expected_checksum}, got {actual_checksum}')
        else:
            logger.debug('All checksums ok')

        font_file = z.open(meta['font']['filename'], 'r')
        model_file = z.open(meta['model']['filename'], 'r')
        font_size = int(meta['image']['font_size'])
        image_size = tuple(meta['image']['image_size'])
        text_location = tuple(meta['image']['text_location'])

        return font_file, model_file, font_size, image_size, text_location
