
class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        # self.writer = tf.summary.FileWriter(log_dir)
        pass

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        pass
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        # self.writer.add_summary(summary, step)
