import unittest

import tensorflow as tf

from network import shuffle_net_v2


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


if __name__ == "__main__":
    unittest.main()
