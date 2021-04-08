import os
import ray

from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()



def restore_saved_model(export_dir):
    signature_key = \
        tf1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    g = tf1.Graph()
    with g.as_default():
        with tf1.Session(graph=g) as sess:
            meta_graph_def = \
                tf1.saved_model.load(sess,
                                     [tf1.saved_model.tag_constants.SERVING],
                                     export_dir)
            print("Model restored!")
            print("Signature Def Information:")
            print(meta_graph_def.signature_def[signature_key])
            print("You can inspect the model using TensorFlow SavedModel CLI.")
            print("https://www.tensorflow.org/guide/saved_model")