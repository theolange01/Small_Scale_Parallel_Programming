Guideline on how to compile and run the kernels on the Crescent environment

This folder contain all the files needed to compile and submit the kernel instructions to the queue. Only the most performing kernels are in theis folde

- Run . env.sh at first
This will set up the environment and load the needed module in order to compile both openMP and CUDA kernels

For the openMP Kernel: 
Go to the openMP directory 
- Run make
This will compile the openMP kernel

The compiled file can then be submit to Crescent CPU Queues using the .sub file

For the CUDA Kernel:
- Run cmake .
This will set up the compiler environment 

Go to the CUDA directory
- Run make
This will compile the CUDA kernel

The compile file can be submit to the Crescent GPU Queue using the .sub file