import tensorflow as tf
import json
import sys
from tensorflow.python.client import timeline
from protobuf_to_dict import protobuf_to_dict
from google.protobuf.json_format import MessageToJson
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import graph_io
import os
import time



dim = 32*(2**9)
dev1 = '/gpu:0'
dev2 = '/gpu:1'
dev3 = '/cpu:0'
logPath = sys.argv[1]


with tf.device(dev1):
    X, Z1, _X = [], [], []
    X.append(tf.random_uniform([dim, dim], 0, 10, name='X' + str(0)))
    _X.append(tf.placeholder(dtype=tf.float32, shape=[dim, dim]))
    Z1.append(tf.matmul(_X[0], _X[0]))

with tf.device(dev2):
    Z3 = []
    Z3.append(tf.matmul(Z1[0], Z1[0]))

with tf.device(dev1):
    Z2 = []
    Z2.append(tf.matmul(Z3[0], Z3[0]))



config_proto = tf.ConfigProto(graph_options=tf.GraphOptions(build_cost_model=1))
config_proto.intra_op_parallelism_threads = 1
config_proto.inter_op_parallelism_threads = 1
config_proto.graph_options.optimizer_options.opt_level = -1
config_proto.graph_options.rewrite_options.constant_folding = (rewriter_config_pb2.RewriterConfig.OFF)
config_proto.graph_options.rewrite_options.arithmetic_optimization = (rewriter_config_pb2.RewriterConfig.OFF)
config_proto.graph_options.rewrite_options.dependency_optimization = (rewriter_config_pb2.RewriterConfig.OFF)
config_proto.graph_options.rewrite_options.layout_optimizer = (rewriter_config_pb2.RewriterConfig.OFF)


sess = tf.Session(config=config_proto)
sess.run(tf.global_variables_initializer())

X_ = sess.run([X])

tot_time = 0
for i in range(10):
    print(i)
    run_metadata = tf.RunMetadata()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
    st = time.time()
    sess.run(Z2,
                {_i: i_ for _i, i_ in zip(_X, X_)},
                options=run_options,
                run_metadata=run_metadata)
    tot_time += time.time() -st

    if i >= 2:
        jsonObj = MessageToJson(run_metadata)
        with open('%s/metadata_matmul_%d.json' % (logPath, i), 'w') as outfile:
            json.dump(jsonObj, outfile)

        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        trace_file = open('%s/timeline_matmul_%d.ctf.json' % (logPath, i), 'w')
        trace_file.write(trace.generate_chrome_trace_format())
print('total time taken : ', tot_time)


