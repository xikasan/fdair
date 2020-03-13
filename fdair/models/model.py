# coding: utf-8

import numpy as np
import tensorflow as tf
tk = tf.keras


class FaultDetector(tk.Model):

    def __init__(
            self,
            units,
            dim_input,
            dim_output,
            name="FaultDetector"
    ):
        super().__init__(name=name)

        assert len(units) == 3
        self.l1 = tk.layers.Dense(units[0], activation="relu", name="L1")
        self.l2 = tk.layers.Dense(units[1], activation="relu", name="L2")
        self.l3 = tk.layers.Dense(units[2], activation="relu", name="L3")
        self.l4 = tk.layers.Dense(dim_output, name="L4")

        dummy_input = tk.Input((dim_input,), dtype=tf.float32)
        self(dummy_input)

    def call(self, inputs):
        feature = self.l1(inputs)
        feature = self.l2(feature)
        feature = self.l3(feature)
        feature = self.l4(feature)
        return feature


if __name__ == '__main__':
    model = FaultDetector(
        [32, 32, 32], 5, 4
    )
    model.summary()
