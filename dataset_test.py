import unittest

import numpy as np
import tensorflow as tf

from dataset import LabelEncoder, load_datasets, preprocess_data
from network import build_model


class TestBoxesFunctions(unittest.TestCase):

    def setUp(self):
        pass

    def test_label_encoder(self):
        d, _ = load_datasets(1)
        for s in d.take(1):
            print(s[0].shape)
            print(s[1].shape)

        encoder = LabelEncoder()
        _,l = encoder.encode_batch(s[0], tf.constant([[0, 1, 0, 1]], shape=[
                             1, 1, 4], dtype=tf.float32), tf.constant([[1]], shape=[1, 1]))
        print(l.shape)

        b_init = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        x = tf.random.normal((s[0].shape))
        model = build_model(2)
        y = model(x)
        print(y.shape)


if __name__ == "__main__":
    unittest.main()
