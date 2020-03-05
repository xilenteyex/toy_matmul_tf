rm -r selCommsRawLogs389376
rm -r selCommsLogs389376



mkdir selCommsRawLogs389376
mkdir selCommsLogs389376

python selectedComms.py 312 selCommsRawLogs389376

cd /root/pesto

python get_graph.py /root/toy_matmul_tf/getComms/selCommsRawLogs389376/metadata_ \
/root/toy_matmul_tf/getComms/selCommsLogs389376/edge-execk

python getKernelCompute.py /root/toy_matmul_tf/getComms/selCommsRawLogs389376/timeline_ \
/root/toy_matmul_tf/getComms/selCommsLogs389376/edge-execk_graph.json \
/root/toy_matmul_tf/getComms/selCommsRawLogs389376/metadata_5.json \
/root/toy_matmul_tf/getComms/selCommsRawLogs389376/metadata_5.json \
/root/toy_matmul_tf/getComms/selCommsLogs389376/edge-newk

python getEdges.py /root/toy_matmul_tf/getComms/selCommsRawLogs389376/metadata_ \
/root/toy_matmul_tf/getComms/selCommsRawLogs389376/timeline_ \
/root/toy_matmul_tf/getComms/selCommsLogs389376/edge-comm



rm -r selCommsRawLogs1444
rm -r selCommsLogs1444



mkdir selCommsRawLogs1444
mkdir selCommsLogs1444

python selectedComms.py 19 selCommsRawLogs1444

cd /root/pesto

python get_graph.py /root/toy_matmul_tf/getComms/selCommsRawLogs1444/metadata_ \
/root/toy_matmul_tf/getComms/selCommsLogs1444/edge-execk

python getKernelCompute.py /root/toy_matmul_tf/getComms/selCommsRawLogs1444/timeline_ \
/root/toy_matmul_tf/getComms/selCommsLogs1444/edge-execk_graph.json \
/root/toy_matmul_tf/getComms/selCommsRawLogs1444/metadata_5.json \
/root/toy_matmul_tf/getComms/selCommsRawLogs1444/metadata_5.json \
/root/toy_matmul_tf/getComms/selCommsLogs1444/edge-newk

python getEdges.py /root/toy_matmul_tf/getComms/selCommsRawLogs1444/metadata_ \
/root/toy_matmul_tf/getComms/selCommsRawLogs1444/timeline_ \
/root/toy_matmul_tf/getComms/selCommsLogs1444/edge-comm




rm -r selCommsRawLogs10227204
rm -r selCommsLogs10227204



mkdir selCommsRawLogs10227204
mkdir selCommsLogs10227204

python selectedComms.py 1599 selCommsRawLogs10227204

cd /root/pesto

python get_graph.py /root/toy_matmul_tf/getComms/selCommsRawLogs10227204/metadata_ \
/root/toy_matmul_tf/getComms/selCommsLogs10227204/edge-execk

python getKernelCompute.py /root/toy_matmul_tf/getComms/selCommsRawLogs10227204/timeline_ \
/root/toy_matmul_tf/getComms/selCommsLogs10227204/edge-execk_graph.json \
/root/toy_matmul_tf/getComms/selCommsRawLogs10227204/metadata_5.json \
/root/toy_matmul_tf/getComms/selCommsRawLogs10227204/metadata_5.json \
/root/toy_matmul_tf/getComms/selCommsLogs10227204/edge-newk

python getEdges.py /root/toy_matmul_tf/getComms/selCommsRawLogs10227204/metadata_ \
/root/toy_matmul_tf/getComms/selCommsRawLogs10227204/timeline_ \
/root/toy_matmul_tf/getComms/selCommsLogs10227204/edge-comm
