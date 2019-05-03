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
    Z1.append(tf.matmul(_X[i], _X[i]))


with tf.device(dev2):
    Y, Z2, _Y = [], [], []
    Y.append(tf.random_uniform([dim, dim], 0, 10, name='Y' + str(0)))
    _Y.append(tf.placeholder(dtype=tf.float32, shape=[dim, dim]))
    Z2.append(tf.matmul(_Y[i], _Y[i]))


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


all_ops = tf.get_default_graph().get_operations()
adj_list_graph_notensors = {}
for op in all_ops:
  adj_list_graph_notensors[op.name] = set([inp.name.split(":")[0] for inp in op.inputs])

adj_list_graph_notensors = {op_name:list(op_deps) for op_name, op_deps in adj_list_graph_notensors.items()}
with open('%s/org_graph_notensors.json' % (logPath), 'w') as outfile:
  json.dump(adj_list_graph_notensors, outfile)

X_, Y_ = sess.run([X, Y])
X_Y_ = X_ + Y_
_X_Y = _X + _Y

tot_time = 0
for i in range(10):
    print(i)
    run_metadata = tf.RunMetadata()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
    st = time.time()
    sess.run(Z1+Z2,
                {_i: i_ for _i, i_ in zip(_X_Y, X_Y_)},
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


