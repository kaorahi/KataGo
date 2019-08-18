import sys
import argparse
import json
import numpy as np
import tensorflow as tf
sys.path.append('../python')
import common
from model import Model

if tf.__version__ != "1.13.1":
    print("use tensorflow 1.13.1 or earlier since TensorFlow 1.14.0 uses BatchMatMulV2 but TensorFlow.js 1.2.6 does not support it yet.")
    sys.exit()

if __name__ == '__main__':
    pos_len = 19
    description = """
    Play go with a trained neural net!
    Implements a basic GTP engine that uses the neural net directly to play moves.
    """

    parser = argparse.ArgumentParser(description=description)
    common.add_model_load_args(parser)
    parser.add_argument('-name-scope', help='Name scope for model variables', required=False)
    args = vars(parser.parse_args())

    (model_variables_prefix, model_config_json) = common.load_model_paths(args)
    name_scope = args["name_scope"]
    with open(model_config_json) as f:
        model_config = json.load(f)

    if name_scope is not None:
        with tf.name_scope(name_scope):
            model = Model(model_config,pos_len,{
                  "is_training": tf.constant(False,dtype=tf.bool),
                  "symmetries": tf.constant(False, shape=[3], dtype=tf.bool),
                  "include_history": tf.constant(1.0, shape=[1,5], dtype=tf.float32)
            })
    else:
        model = Model(model_config,pos_len,{
                  "is_training": tf.constant(False,dtype=tf.bool),
                  "symmetries": tf.constant(False, shape=[3], dtype=tf.bool),
                  "include_history": tf.constant(1.0, shape=[1,5], dtype=tf.float32)
        })

    saver = tf.train.Saver(
        max_to_keep = 10000,
        save_relative_paths = True,
    )

    with tf.Session() as session:
        saver.restore(session, model_variables_prefix)
        tf.compat.v1.saved_model.simple_save(
            session,
            './saved_model',
            inputs={
                "swa_model/bin_inputs": model.bin_inputs,
                "swa_model/global_inputs": model.global_inputs,
            },
            outputs={
                'swa_model/policy_output': model.policy_output,
                'swa_model/value_output': model.value_output,
                'swa_model/miscvalues_output': model.miscvalues_output,
                'sswa_model/corebelief_output': model.scorebelief_output,
                'swa_model/bonusbelief_output': model.bonusbelief_output,
                'swa_model/ownership_output': model.ownership_output,
            }
        )
