rm -r allCommsRawLogs
rm -r allCommsLogs



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



cd /root/toy_matmul_tf/getComms

mkdir allCommsG2RawLogs
mkdir allCommsG2Log

python allCommsG2.py allCommsG2RawLogs

cd /root/pesto

python get_graph.py /root/toy_matmul_tf/getComms/allCommsG2RawLogs/metadata_ \
/root/toy_matmul_tf/getComms/allCommsG2Logs/edge-execk

python getKernelCompute.py /root/toy_matmul_tf/getComms/allCommsG2RawLogs/timeline_ \
/root/toy_matmul_tf/getComms/allCommsG2Logs/edge-execk_graph.json \
/root/toy_matmul_tf/getComms/allCommsG2RawLogs/metadata_5.json \
/root/toy_matmul_tf/getComms/allCommsG2RawLogs/metadata_5.json \
/root/toy_matmul_tf/getComms/allCommsG2Logs/edge-newk

python getEdges.py /root/toy_matmul_tf/getComms/allCommsG2RawLogs/metadata_ \
/root/toy_matmul_tf/getComms/allCommsG2RawLogs/timeline_ \
/root/toy_matmul_tf/getComms/allCommsG2Logs/edge-comm


