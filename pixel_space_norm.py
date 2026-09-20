"""This module contains functions for normalizing the pixel spacing of the images."""

import numpy as np
import cv2 as cv
from numpy.lib.stride_tricks import sliding_window_view


def median_5x5_field_value(image):
    """Return the median average value of all valid 5x5 fields."""
    if min(image.shape[:2]) < 5:
        return np.median(image, axis=(0, 1))

    fields = sliding_window_view(image, (5, 5), axis=(0, 1))
    field_averages = fields.mean(axis=(-2, -1))
    return np.median(field_averages, axis=(0, 1))


def normalize_inplane_fov(images, pixel_spacing_mm, output_shape=(256, 256), target_fov_mm=500):
    """Make image pixels represent the same field of view."""
    spacing_x, spacing_y = pixel_spacing_mm
    target_shape = (round(target_fov_mm / spacing_x), round(target_fov_mm / spacing_y))
    normalized = []

    for image in images:
        channels = image.shape[2:] if image.ndim > 2 else ()
        padding_value = np.asarray(median_5x5_field_value(image), dtype=image.dtype)
        canvas = np.empty((*target_shape, *channels), dtype=image.dtype)
        canvas[...] = padding_value
        source_slices = []
        target_slices = []

        for source_size, target_size in zip(image.shape[:2], target_shape):
            offset = (source_size - target_size) // 2
            source_start = max(offset, 0)
            source_stop = min(offset + target_size, source_size)
            target_start = max(-offset, 0)
            target_stop = target_start + source_stop - source_start
            source_slices.append(slice(source_start, source_stop))
            target_slices.append(slice(target_start, target_stop))

        canvas[tuple(target_slices) + (slice(None),) * len(channels)] = image[
            tuple(source_slices) + (slice(None),) * len(channels)
        ]
        normalized.append(cv.resize(canvas, output_shape[::-1], interpolation=cv.INTER_LINEAR))

    return normalized