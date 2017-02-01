# ConvolutionNeuralNet
Original LeNet-5 and Modified CNN with improvement on shift and background invariance

Note: Lua code with Torch library

Programs: 
1) LeNet5.lua - Implementation of LeNet-5 with mean-average percision (mAP) of 99.19% if set learning rate = 0.o3, #Epochs= 30 and batchSize=30.
2) modified CNN.lua - The original LeNet-5 does not perform well with background and shift variance.The modified CNN has an additional convolution layer and overlapping subsampling. 
                  It yields 10.6% higher mAP than the original LeNet-5 on a randomly shifted testset.  
