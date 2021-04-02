import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import softmax
import numpy as np
import tensorflow as tf

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
    
def identity(x):
    return x
class PredictionPlot(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, name, dataset, label_transform=None):
        self.log_dir = log_dir
        self.dataset = dataset
        self.name = name
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.transform = label_transform or identity
        super(PredictionPlot, self).__init__()
    def get_plot(self):
        ds = self.dataset.as_numpy_iterator()
        fig, ax = plt.subplots(1)
        values = []
        labels = []
        for batch in ds:
            labels += self.transform(batch[1]).tolist()
            vals = self.model.predict_on_batch(batch[0])
            #vals = softmax(vals, axis=1)
            values += self.transform(vals).tolist()
        ax.scatter(values, labels)
        ax.set_aspect('equal')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        return fig
    def on_epoch_end(self, epoch, logs=None):
        image = plot_to_image(self.get_plot())
        with self.file_writer.as_default():
            tf.summary.image(self.name, image, step=epoch)