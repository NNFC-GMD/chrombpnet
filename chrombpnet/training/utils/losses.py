import tensorflow as tf


#from https://github.com/kundajelab/basepair/blob/cda0875571066343cdf90aed031f7c51714d991a/basepair/losses.py#L87
def multinomial_nll(true_counts, logits):
    """Compute the multinomial negative log-likelihood
    Args:
      true_counts: observed count values
      logits: predicted logit values
    """
    true_counts = tf.cast(true_counts, tf.float32)
    logits = tf.cast(logits, tf.float32)
    counts_per_example = tf.reduce_sum(true_counts, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    log_likelihood = (
        tf.math.lgamma(counts_per_example + 1.0)
        - tf.reduce_sum(tf.math.lgamma(true_counts + 1.0), axis=-1)
        + tf.reduce_sum(true_counts * log_probs, axis=-1)
    )
    return -tf.reduce_mean(log_likelihood)



