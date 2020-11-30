import unittest

import numpy as np
import tensorflow as tf

from network import (build_head, build_retinanet, feature_pyramid,
                     get_backbone, shuffle_net_v2)


class TestBoxesFunctions(unittest.TestCase):

    def setUp(self):
        pass

    def test_feature_map_shape(self):
        intput_shape = [None, None, 3]
        x = tf.random.normal((1, 224, 224, 3))
        y = shuffle_net_v2(intput_shape)(x)
        self.assertListEqual([1, 28, 28, 116], y[0].shape.as_list())
        self.assertListEqual([1, 14, 14, 232], y[1].shape.as_list())
        self.assertListEqual([1, 7, 7, 464], y[2].shape.as_list())

    def test_fpn(self):
        x = tf.random.normal((1, 224, 224, 3))
        b = get_backbone()
        p2, p3, p4 = feature_pyramid(b)(x)
        self.assertListEqual([1, 28, 28, 256], p2.shape.as_list())
        self.assertListEqual([1, 14, 14, 256], p3.shape.as_list())
        self.assertListEqual([1, 7, 7, 256], p4.shape.as_list())

    def test_build_head(self):
        b_init = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        x = tf.random.normal((1, 224, 224, 3))
        b = get_backbone()
        p2, p3, p4 = feature_pyramid(b)(x)
        cls_head = build_head(9*2, b_init)
        box_head = build_head(9*4, b_init)

        c2 = cls_head(p2)
        b2 = box_head(p2)

        c3 = cls_head(p3)
        b3 = box_head(p3)

        c4 = cls_head(p4)
        b4 = box_head(p4)

        self.assertListEqual([1, 28, 28, 9*2], c2.shape.as_list())
        self.assertListEqual([1, 28, 28, 9*4], b2.shape.as_list())
        self.assertListEqual([1, 14, 14, 9*2], c3.shape.as_list())
        self.assertListEqual([1, 14, 14, 9*4], b3.shape.as_list())
        self.assertListEqual([1, 7, 7, 9*2], c4.shape.as_list())
        self.assertListEqual([1, 7, 7, 9*4], b4.shape.as_list())

    def test_build_model(self):
        b_init = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        x = tf.random.normal((1, 224, 224, 3))
        model = build_retinanet(2, get_backbone())
        y = model(x)

        self.assertListEqual([1, (28*28+14*14+7*7)*9, 4+2], y.shape.as_list())


if __name__ == "__main__":
    unittest.main()
