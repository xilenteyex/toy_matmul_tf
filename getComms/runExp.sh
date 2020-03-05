mkdir allCommsRawLogs
mkdir allCommsLogs

python allComms.py allCommsRawLogs

cd /root/pesto

python get_graph.py /root/toy_matmul_tf/getComms/allCommsRawLogs/metadata_ \
/root/toy_matmul_tf/getComms/allCommsLogs/edge-execk

python getKernelCompute.py /root/toy_matmul_tf/getComms/allCommsRawLogs/timeline_ \
/root/toy_matmul_tf/getComms/allCommsLogs/edge-execk_graph.json \
/root/toy_matmul_tf/getComms/allCommsRawLogs/metadata_5.json \
/root/toy_matmul_tf/getComms/allCommsRawLogs/metadata_5.json \
/root/toy_matmul_tf/getComms/allCommsLogs/edge-newk

python getEdges.py /root/toy_matmul_tf/getComms/allCommsRawLogs/metadata_ \
/root/toy_matmul_tf/getComms/allCommsRawLogs/timeline_ \
/root/toy_matmul_tf/getComms/allCommsLogs/edge-comm
