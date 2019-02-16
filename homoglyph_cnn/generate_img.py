# -*- coding: utf-8 -*-


from functools import lru_cache
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont


@lru_cache(maxsize=32)
def _load_font(font_location: str, font_size: int) -> FreeTypeFont:
    return ImageFont.truetype(font=font_location, size=font_size)


@lru_cache(maxsize=128)
def generate_img(string: str, font_location: str, font_size: int, image_size: Tuple[int, int], text_location: Tuple[int, int]) -> np.ndarray:
    """
    Generates an image with the given text

    :param string: the text to draw
    :type string: str
    :param font_location: the path to the font
    :type font_location: str
    :param font_size: the size of the font
    :type font_size: int
    :param image_size: the size of the image
    :type image_size: Tuple[int, int]
    :param text_location: the starting location of the text
    :type text_location: Tuple[int, int]
    :return: the image with the text drawn on
    :rtype: np.ndarray
    """

    # load font
    font = _load_font(font_location=font_location, font_size=font_size)

    # create image
    img = Image.new(mode='F', size=image_size)

    # draw text on image
    dimg = ImageDraw.Draw(im=img)
    dimg.text(xy=text_location, text=string.lower(), font=font)

    return np.expand_dims(a=img, axis=0)
