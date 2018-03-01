# Homework 4

### Samuel Badman - 02/28/18

I'm submitting my homework as a jupter notebook :
 hw4_notebook_badman.ipynb
 
It should work to clear output and run all cells, and the plot should appear 
after the last python cell. My explanation/discussion of the plots are in a 
markdown cell at the end of the notebook. The text of this is:


__Conclusions__

My machine is a Linux machine with 8 cores on an Intel 2.8 GHz Chip 

We can see the same heuristic features as in the example plots in the homework pdf : the __parallelization is only advantageous as the number of darts thrown gets very large__. At small dart throws, the simple serial implementation is much faster as the __parallelization methods require overhead time to set up communication__ between processes / cores. We see that as the number of cores increases, the efficiency of the multiprocessing technique surpasses the serial technique at around 1,000 darts, and the IPcluster method catches up soon after at around 100,000 darts. Beyond 1,000,000 darts, the IPCluster and Multiprocessing methods converge to around the same time. For all three techniques, the behaviour __at large numbers of darts is roughly linear in log-log space__ (exponential growth in linear space), but the parallel techniques log-log linear graph has a significantly lower execution time intercept, meaning the real time run times are much lower when we make use of parallelization. For example, at $10^7$ darts, the serial method takes around 14s to run on my machine, compared to around 4s with the two parallelized methods. This is because I give the program N tasks to do, and the __parallel techniques allow N/number of processes to be computed simultaneously__. In the limit where the time for completion in serial is much greater than the overhead time required for the parallel processes to communicate with each other, this means the parallelized times are much faster.

Comparing to the example plots, there is some discrepancy between the performance of IPCluster at the lowest dart numbers (N=10) - on my computer this consistently takes around 10x longer than the examples. The multiprocessing is also fractionally slower at low dart numbers, but otherwise there is very good agreement between my plot and the example, especially as number of darts gets very large. I'm not sure why this discrepancy in IPCluster run time exists but given where in my code I am timing I know it is not due to setting up the clusters or instantiating the client, so the difference must be happening in the map_sync function, and given the low computational requirement with 10 darts, the time is presumably coming from communicating with the engines.


