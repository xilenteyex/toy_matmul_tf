# array=( 9 10 12 14 16 19 22 26 31 37 44 52 62 74 88 105 126 151 181 217 260 312 374 448 537 644 772 926 1111 1333 1599 1918 2301 2761 3313 3975 4770 5724 6868 8241 9889 11866 14239 17086 20503 )
array=( 9 10 )
for i in "${array[@]}"
do
	rm -r selCommsRawLogs$i
	rm -r selCommsLogs$i

	mkdir selCommsRawLogs$i
	mkdir selCommsLogs$i

	python selectedComms.py $i selCommsRawLogs$i

	cd /root/pesto

	python get_graph.py /root/toy_matmul_tf/getComms/selCommsRawLogs$i/metadata_ \
	/root/toy_matmul_tf/getComms/selCommsLogs$i/edge-execk

	python getKernelCompute.py /root/toy_matmul_tf/getComms/selCommsRawLogs$i/timeline_ \
	/root/toy_matmul_tf/getComms/selCommsLogs$i/edge-execk_graph.json \
	/root/toy_matmul_tf/getComms/selCommsRawLogs$i/metadata_5.json \
	/root/toy_matmul_tf/getComms/selCommsRawLogs$i/metadata_5.json \
	/root/toy_matmul_tf/getComms/selCommsLogs$i/edge-newk

	python getEdges.py /root/toy_matmul_tf/getComms/selCommsRawLogs$i/metadata_ \
	/root/toy_matmul_tf/getComms/selCommsRawLogs$i/timeline_ \
	/root/toy_matmul_tf/getComms/selCommsLogs$i/edge-comm


	cd /root/toy_matmul_tf/getComms
   # do whatever on $i
done