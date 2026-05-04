import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="chrombpnet")
class LogcountSum(Layer):
    """Combine sequence and bias log-count heads with log-sum-exp."""

    def call(self, inputs):
        return tf.math.reduce_logsumexp(inputs, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (1,)
