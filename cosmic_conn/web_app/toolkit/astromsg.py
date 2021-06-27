# -*- coding: utf-8 -*-

"""
Define payload classes for data transmission between front-end and back-end.
Payload Types should be consistent between the backend and the front end
Boning Dong, CY Xu (cxu@ucsb.edu)
"""

import numpy as np


class PayloadTypes:
    NONE_TYPE = 0
    IMAGE_TYPE = 1
    THUMBNAIL_TYPE = 2
    FLOATLIST_TYPE = 3


"""
Payload format:
[payload size: uint32][payload type: uint8][payload]
note: the payload size doesn't include payload size and the payload type field
"""


def add_payload_meta_and_to_bytes(payload):
    payload_bytes: bytes = payload.tobytes()
    payload_size = np.array([len(payload_bytes)], dtype=np.uint32)
    payload_type = np.array([payload.get_payload_type()], dtype=np.uint8)
    payload_bytes = payload_size.tobytes() + payload_type.tobytes() + payload_bytes
    return payload_bytes


"""
PostResponse format
[Payload Count: uint32][ payload 1 ][ payload 2 ]
"""


class PostResponse:
    def __init__(self):
        self.payloads = []

    def append_payload(self, payload):
        self.payloads.append(payload)

    def tobytes(self):
        payload_count = np.array([len(self.payloads)], dtype=np.uint32)
        response_bytes = payload_count.tobytes()
        for payload in self.payloads:
            response_bytes += add_payload_meta_and_to_bytes(payload)
        return response_bytes


"""
ImagePayload format
[width: uint32][height: uint32][ image array: float32 ]
"""


class ImagePayload:
    def __init__(self, img_2d_array):
        h, w = img_2d_array.shape
        self.img_height = h
        self.img_width = w
        self.img_2d_array = img_2d_array

    def get_payload_type(self):
        return PayloadTypes.IMAGE_TYPE

    def tobytes(self):
        img_meta = np.array([self.img_width, self.img_height], dtype=np.uint32)
        img_2d_array_float32 = self.img_2d_array.astype("float32")
        img_payload_bytes = img_meta.tobytes() + img_2d_array_float32.tobytes()
        return img_payload_bytes


"""
ThumbnailPayload format
[thumbnail number: uint32][patch size: uint32][coordinate array: x: uint32, y: uint32, x:uint32, y:uint32 ...]
"""


class ThumbnailPayload:
    def __init__(self, patch_size):
        self.patch_size = patch_size
        self.thumbnail_coords = []

    def add_thumbnail_coord(self, coord):
        self.thumbnail_coords.append(coord)

    def get_payload_type(self):
        return PayloadTypes.THUMBNAIL_TYPE

    def tobytes(self):
        thumbnail_meta = np.array(
            [len(self.thumbnail_coords), self.patch_size], dtype=np.uint32
        )
        thumbnail_body = np.array(
            self.thumbnail_coords).flatten().astype(np.uint32)
        thumbnail_payload_bytes = thumbnail_meta.tobytes() + thumbnail_body.tobytes()
        return thumbnail_payload_bytes


"""
Floating Point List format
[length of list: uint32][list of data in float32: [zscale z1, zscale z2, ...]]
"""


class FloatListPayload:
    def __init__(self, list_of_data):
        # self.length = len(list_of_data)
        self.data = list_of_data

    def get_payload_type(self):
        return PayloadTypes.FLOATLIST_TYPE

    def tobytes(self):
        # length = np.array(self.length, dtype=np.uint32).tobytes()
        zscale = np.array(self.data[:2], dtype="float32").tobytes()
        # length +
        return zscale
