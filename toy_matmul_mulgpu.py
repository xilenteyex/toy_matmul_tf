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

n = 9
dim = 32
Z1, Z2, W = [], [], []
X, _X, Y, _Y = [], [], [], []
r = float(sys.argv[1])


dim = 32
n = 9
op = 'mul'
dev1 = '/gpu:0'
dev2 = '/gpu:1'
dev3 = '/cpu:0'
r = float(sys.argv[1])


with tf.device(dev1):
    X, Z1, _X = [], [], []
    for i in range(n):
        dim = int(r*dim)
        X.append(tf.random_uniform([dim, dim], 0, 10, name='X' + str(i)))
        _X.append(tf.placeholder(dtype=tf.float32, shape=[dim, dim]))
        Z1.append(tf.matmul(_X[i], _X[i]))

dim = 32

with tf.device(dev1):
    Y, Z2, _Y = [], [], []
    for i in range(n):
        dim = int(r*dim)
        Y.append(tf.random_uniform([dim, dim], 0, 10, name='Y' + str(i)))
        _Y.append(tf.placeholder(dtype=tf.float32, shape=[dim, dim]))
        Z2.append(tf.matmul(_Y[i], _Y[i]))

count = 0

with tf.device(dev1):
    W1 = []
    for j in range(n):
        if (j % 2 == 0):
            W1.append(tf.matmul(Z1[j], Z2[j]))
            count += 1

with tf.device(dev2):
    W2 = []
    for j in range(n):
        if (j % 2 != 0):
            W2.append(tf.matmul(Z1[j], Z2[j]))
            count += 1
print('matmul count : ', count)

with tf.device(dev3):
    W3 = []
    for w in W1:
        W3.append(tf.add(w, w))
    for w in W2:
        W3.append(tf.add(w, w))


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
with open('commLogs/org_graph_notensors_%.2f.json' % (r), 'w') as outfile:
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
    W_ = sess.run(W3,
                {_i: i_ for _i, i_ in zip(_X_Y, X_Y_)},
                options=run_options,
                run_metadata=run_metadata)
    tot_time += time.time() -st

    if i >= 2:
        jsonObj = MessageToJson(run_metadata)
        with open('commLogs/metadata_matmul_%.2f_%d.json' % (r, i), 'w') as outfile:
            json.dump(jsonObj, outfile)

        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        trace_file = open('commLogs/timeline_matmul_%.2f_%d_.ctf.json' % (r, i), 'w')
        trace_file.write(trace.generate_chrome_trace_format())
print('total time taken : ', tot_time)


