import tensorflow as tf
import tensorflow.contrib.tfprof as tfprof
from tensorflow.python.client import timeline
from protobuf_to_dict import protobuf_to_dict
from google.protobuf.json_format import MessageToJson
import json
import sys
import numpy as np
from tensorflow.core.protobuf import rewriter_config_pb2
from toposort import toposort
import time

from utils import read_json_file
import json

# import urllib2
# response = urllib2.urlopen("https://raw.githubusercontent.com/yaroslavvb/memory_util/master/memory_util.py")
# open("memory_util.py", "wb").write(response.read())

# import memory_util
# memory_util.vlog(1)

# dev_map = None
ops_placed = 0
ops_tot = 0
dev_map = read_json_file('toy_matmul_map.json')

def test_device_placer(op):
    global ops_placed, ops_tot
    ops_tot += 1

    if op.name in dev_map:
        ops_placed += 1
        return dev_map[op.name]

    return op.device


# with tf.device(test_device_placer):

dim = 32
n = 9
op = 'mul'
dev1 = '/gpu:0'
dev2 = '/gpu:1'

with tf.device(dev1):
    X, Z1, _X = [], [], []
    for i in range(n):
        dim *= 2
        X.append(tf.random_uniform([dim, dim], 0, 10, name='X' + str(i)))
        _X.append(tf.placeholder(dtype=tf.float32, shape=[dim, dim]))
        Z1.append(tf.matmul(_X[i], _X[i]))

dim = 32

with tf.device(dev1):
    Y, Z2, _Y = [], [], []
    for i in range(n):
        dim *= 2
        Y.append(tf.random_uniform([dim, dim], 0, 10, name='Y' + str(i)))
        _Y.append(tf.placeholder(dtype=tf.float32, shape=[dim, dim]))
        Z2.append(tf.matmul(_Y[i], _Y[i]))

count = 0

W1 = []
W2 = []

with tf.device(dev1):
    for j in range(n):
        # if (j % 2 == 0):
        if op == 'mul':
            W1.append(tf.matmul(Z1[j], Z2[j]))
            count += 1
        elif op == 'add':
            W1.append(tf.add(Z1[j], Z2[j]))

# with tf.device(dev1):
#     for j in range(n):
#         # if (j % 2 != 0):
#         if op == 'mul':
#             W2.append(tf.matmul(Z1[j], Z2[j]))
#             count += 1
#         elif op == 'add':
#             W2.append(tf.add(Z1[j], Z2[j]))

config_proto = tf.ConfigProto(graph_options=tf.GraphOptions(build_cost_model=1))
config_proto.intra_op_parallelism_threads = 1
config_proto.inter_op_parallelism_threads = 1
config_proto.graph_options.optimizer_options.opt_level = -1
config_proto.graph_options.rewrite_options.constant_folding = (rewriter_config_pb2.RewriterConfig.OFF)
config_proto.graph_options.rewrite_options.arithmetic_optimization = (rewriter_config_pb2.RewriterConfig.OFF)


sess = tf.Session(config=config_proto)
sess.run(tf.global_variables_initializer())
print("initialized")

for i in range(2):
    print(i)
    X_, Y_ = sess.run([X, Y])

    # run_metadata = tf.RunMetadata()
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)

    st_time = time.time()

    # W_ = sess.run(W1+W2,
    #               {_i: i_ for _i, i_ in zip(_X + _Y, X_ + Y_)},
    #               options=run_options,
    #               run_metadata=run_metadata)

    W_ = sess.run(W1+W2,
                  {_i: i_ for _i, i_ in zip(_X + _Y, X_ + Y_)})

    en_time = time.time() - st_time
    print('Total Time : ', en_time, ' seconds')
    print('Ops placed : ', ops_placed)
    print('Ops Total : ', ops_tot)

    
    # jsonObj = MessageToJson(run_metadata)
    # with open('metadata_only1gputest_%d.json' % (i), 'w') as outfile:
    #     json.dump(jsonObj, outfile)

    # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
    # trace_file = open('timeline_only1gputest_%d.ctf.json' % (i), 'w')
    # trace_file.write(trace.generate_chrome_trace_format())

