import sys, os
import json
import tensorflow as tf
from absl import logging
from athena import DecoderSolver
from athena.main import (
    parse_config,
    build_model_from_jsonfile,
    SUPPORTED_DATASET_BUILDER
)
from pydecoders import WFSTDecoder

def decode(jsonfile, n=1, log_file=None):
    """ entry point for model decoding use athena decoder, do some preparation work """
    p, model, _, checkpointer = build_model_from_jsonfile(jsonfile)
    checkpoint_wer_dict = {}
    for line in open('screenlog.0'):
        if 'epoch:' in line:
            splits = line.strip().split('\t')
            epoch = int(splits[0].split(' ')[-1])
            ctc_acc = float(splits[-1].split(' ')[-1])
            checkpoint_wer_dict[epoch] = ctc_acc
    checkpoint_wer_dict = {k: v for k, v in sorted(checkpoint_wer_dict.items(), key=lambda item: item[1], reverse=True)}
    ckpt_index_list = list(checkpoint_wer_dict.keys())[0: n]
    print('best_wer_checkpoint: ')
    print(ckpt_index_list)
    ckpt_v_list = []

    #restore v from ckpts
    for idx in ckpt_index_list:
        ckpt_path = p.ckpt + 'ckpt-' + str(idx)
        checkpointer.restore(ckpt_path) #current variables will be updated
        var_list = []
        for i in model.trainable_variables:
            v = tf.constant(i.value())
            var_list.append(v)
        ckpt_v_list.append(var_list)
    #compute average, and assign to current variables
    for i in range(len(model.trainable_variables)):
        v = [tf.expand_dims(ckpt_v_list[j][i],[0]) for j in range(len(ckpt_v_list))]
        v = tf.reduce_mean(tf.concat(v,axis=0),axis=0)
        model.trainable_variables[i].assign(v)
    solver = DecoderSolver(model, config=p.athena_decoder)
    assert p.athena_decoder is not None
    dataset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.testset_config)
    dataset_builder = dataset_builder.compute_cmvn_if_necessary(True)
    solver.decode_use_athena_decoder(dataset_builder.as_dataset(batch_size=1))

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 2:
        logging.warning('Usage: python {} config_json_file'.format(sys.argv[0]))
        sys.exit()
    tf.random.set_seed(1)

    jsonfile = sys.argv[1]
    with open(jsonfile) as file:
        config = json.load(file)
    p = parse_config(config)
    #DecoderSolver.initialize_devices(p.solver_gpu)
    decode(jsonfile, n=10, log_file='screenlog.0')


