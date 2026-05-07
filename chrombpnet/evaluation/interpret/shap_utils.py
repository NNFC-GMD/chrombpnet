# Av's code with a bit of reformatting
# Adapted from Zahoor's mtbatchgen

import numpy as np
import shap
import tensorflow as tf
from tensorflow.keras.layers import Lambda

from deeplift.dinuc_shuffle import dinuc_shuffle


def create_tf1_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)


def require_tf_graph_tensor(tensor, tensor_name):
    if not hasattr(tensor, "op"):
        raise RuntimeError(
            "DeepSHAP requires TensorFlow graph tensors from legacy tf.keras. "
            "Set TF_USE_LEGACY_KERAS=1 before importing TensorFlow and install "
            "tf-keras==2.21.0 in the DeepSHAP environment. "
            "{} is a {} instead.".format(tensor_name, type(tensor).__name__)
        )


def combine_mult_and_diffref(mult, orig_inp, bg_data):
    to_return = []
    
    for l in [0]:
        projected_hypothetical_contribs = \
            np.zeros_like(bg_data[l]).astype("float")
        assert len(orig_inp[l].shape)==2
        
        # At each position in the input sequence, we iterate over the
        # one-hot encoding possibilities (eg: for genomic sequence, 
        # this is ACGT i.e. 1000, 0100, 0010 and 0001) and compute the
        # hypothetical difference-from-reference in each case. We then 
        # multiply the hypothetical differences-from-reference with 
        # the multipliers to get the hypothetical contributions. For 
        # each of the one-hot encoding possibilities, the hypothetical
        # contributions are then summed across the ACGT axis to 
        # estimate the total hypothetical contribution of each 
        # position. This per-position hypothetical contribution is then
        # assigned ("projected") onto whichever base was present in the
        # hypothetical sequence. The reason this is a fast estimate of
        # what the importance scores *would* look like if different 
        # bases were present in the underlying sequence is that the
        # multipliers are computed once using the original sequence, 
        # and are not computed again for each hypothetical sequence.
        for i in range(orig_inp[l].shape[-1]):
            hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
            hypothetical_input[:, i] = 1.0
            hypothetical_difference_from_reference = \
                (hypothetical_input[None, :, :] - bg_data[l])
            hypothetical_contribs = hypothetical_difference_from_reference * \
                                    mult[l]
            projected_hypothetical_contribs[:, :, i] = \
                np.sum(hypothetical_contribs, axis=-1) 
            
        to_return.append(np.mean(projected_hypothetical_contribs,axis=0))

    if len(orig_inp)>1:
        to_return.append(np.zeros_like(orig_inp[1]))
    
    return to_return


def shuffle_several_times(s,numshuffles=20):
    if len(s)==2:
        return [np.array([dinuc_shuffle(s[0]) for i in range(numshuffles)]),
                np.array([s[1] for i in range(numshuffles)])]
    else:
        return [np.array([dinuc_shuffle(s[0]) for i in range(numshuffles)])]


def _sum_last_axis(x):
    return tf.reduce_sum(x, axis=-1)


def get_counts_output(model):
    return Lambda(_sum_last_axis,
                  output_shape=(),
                  name="shap_sum_logcount_predictions")(model.outputs[1])


def _weightedsum_meannormed_logits(logits):
    meannormed_logits = logits - tf.reduce_mean(logits, axis=1, keepdims=True)

    # 'stop_gradient' will prevent importance from being propagated
    # through this operation; we do this because we just want to treat
    # the post-softmax probabilities as 'weights' on the different
    # logits, without having the network explain how the probabilities
    # themselves were derived. Could be worth contrasting explanations
    # derived with and without stop_gradient enabled...
    stopgrad_meannormed_logits = tf.stop_gradient(meannormed_logits)
    softmax_out = tf.nn.softmax(stopgrad_meannormed_logits, axis=1)

    # Weight the logits according to the softmax probabilities, take
    # the sum for each example. This mirrors what was done for the
    # bpnet paper.
    return tf.reduce_sum(softmax_out * meannormed_logits, axis=1)


def get_weightedsum_meannormed_logits(model):
    # Assumes the 0 task track is for profile
    # See Google slide deck for explanations
    # We meannorm as per section titled 
    # "Adjustments for Softmax Layers" in the DeepLIFT paper
    return Lambda(_weightedsum_meannormed_logits,
                  output_shape=(),
                  name="shap_weightedsum_meannormed_logits")(model.outputs[0])
