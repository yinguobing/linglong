import unittest

import tensorflow as tf

from network import shuffle_net_v2, get_backbone, feature_pyramid


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




if __name__ == "__main__":
    unittest.main()
