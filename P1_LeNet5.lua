--Karen Wang
--wangkh@usc.edu
--Fall 2016

--LeNet Implementation
--2+1 Stage CNN
--Original Testset

require "nn"
require "optim"
require "image"

function main()
  -- Fix the random seed for debugging.
  torch.manualSeed(0)
  
-------------------- 1. PROCESS INPUT DATA --------------------
--a) Load Dataset
  trainset = torch.load('mnist-p1b-train.t7')
  testset = torch.load('mnist-p1b-test.t7')
  classes= {'0','1','2','3','4','5','6','7','8','9'}
  --[[debug:]]
  --print(trainset)
  --print(testset)
  --print(#trainset.data)
  --print(classes[trainset.label[100]])

--b) Set-up Meta-table
  setmetatable(trainset, 
      {__index = function(t, i) 
                      return {t.data[i], t.label[i]} 
                  end}
                  );

 --c) Define helper functions
  trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

  function trainset:size() 
      return self.data:size(1) 
  end

 --[[debug:]]
 --print(trainset:size())
 --print('Trainset Size:',trainset:size())


-------------------- 2.1 PRE-PROCESSING of TRAIN DATA (zero mean, normalized std) --------------------
  mean = {} -- store the mean, to normalize the test set in the future
  stdv  = {} -- store the standard-deviation for the future
  for i=1,1 do -- over each image channel
      mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
      print('Channel ' .. i .. ', Mean: ' .. mean[i])
      trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
      
      stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
      print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
      trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
  end

 -- Save to .png
  do
  image.save('patterns.png',image.toDisplayTensor(trainset.data:narrow(1,1,20),1,10))
  end


  -------------------- 3. CREATE NETWORK --------------------
  -- Create the network and the criterion.
  print('Creating the network and the criterion...')

  local network   = nn.Sequential()

--a) 1st Stage
  -- A view layer so the network recognize a batch. (View layers do not perform any computation. They tell the network how to look at the data.)
  network:add(nn.View(1,32,32):setNumInputDims(3))
  
   -- 1 input image channel, 6 output channels, 5x5 convolution kernel
  network:add(nn.SpatialConvolution(1, 6, 5, 5)) -- 1x32x32->6x28x28

  -- A ReLU activation layer.
  network:add(nn.ReLU())

  -- A maxpooling layer.
  network:add(nn.SpatialMaxPooling(2,2,2,2)) -- 6x28x28->16x10x10
  
--b) 2nd Stage
  network:add(nn.SpatialConvolution(6, 16, 5, 5))
  network:add(nn.ReLU()) 
  network:add(nn.SpatialMaxPooling(2,2,2,2))
  
  -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
  network:add(nn.View(16*5*5))
  
--c) Fully connected layer (matrix multiplication between input and weights)
  network:add(nn.Linear(16*5*5, 120))             
  network:add(nn.ReLU()) -- non-linearity 
  network:add(nn.Linear(120, 84))
  network:add(nn.ReLU()) -- non-linearity 
  
  -- 10 is the number of outputs of the network (in this case, 10 digits)
  network:add(nn.Linear(84, 10)) 
  
  -- converts the output to a log-probability. Useful for classification problems
  network:add(nn.LogSoftMax()) 
  
  print('Lenet5\n' .. network:__tostring());


-------------------- 4. TRAIN NETWORK --------------------   
local criterion = nn.ClassNLLCriterion()

--Method#1: NN Stochastic Gradient Trainer -- (worse performance)
  --[[trainer = nn.StochasticGradient(network, criterion)
  trainer.learningRate = 0.001 --0.001
  trainer.maxIteration = 20 -- just do 5 epochs of training
  trainer:train(trainset)]]--

-- Method#2: Optim SGD Method
  -- Extract the parameters and arrange them linearly in memory,so we have a large vector containing all the parameters.
  parameters,gradParameters = network:getParameters()
  
  print('Training...')

  -- Hyperparameters
  local nEpoch = 30
  for e = 1,nEpoch do
    -- Number of training samples.
    local size  = trainset.data:size()[1]
    -- Batch size. We use a batch of samples to "smooth" the gradients.
    local bsize = 30
    -- Total loss.
    local tloss = 0

    -- Confusion matrix.
    local confusion = optim.ConfusionMatrix(classes)
    for t  = 1,size,bsize do
      local bsize = math.min(bsize,size-t+1)
      local input  = trainset.data:narrow(1,t,bsize)
      local target = trainset.label:narrow(1,t,bsize)
      -- Reset the gradients to zero.
      gradParameters:zero()
      -- Forward the data and compute the loss.
      local output = network:forward(input)
      local loss   = criterion:forward(output,target)
      -- Collect Statistics
      tloss = tloss + loss * bsize
      confusion:batchAdd(output,target)
      -- Backward. The gradient wrt the parameters are internally computed.
      local gradOutput = criterion:backward(output,target)
      local gradInput  = network:backward(input,gradOutput)
      -- The optim module accepts a function for evaluation.
      -- For simplicity I made the computation outside, and
      -- this function is used only to return the result.
      local function feval()
        return loss,gradParameters
      end
      
      -- Specify the training parameters.
      local config = {
        learningRate = 0.03,
      }
      -- We use the SGD method.
      optim.sgd(feval, parameters, config)
      -- Show the progress.
      io.write(string.format("progress: %4d/%4d\r",t,size))
      io.flush()
    end
    -- Compute the average loss.
    tloss = tloss / size
    -- Update the confusion matrix.
    confusion:updateValids()
    -- Let us print the loss and the accuracy.
    -- You should see the loss decreases and the accuracy increases as the training progresses.
    print(string.format('epoch = %2d/%2d  loss = %.4f accuracy = %.2f',e,nEpoch,tloss,100*confusion.totalValid))
    -- You can print the confusion matrix if you want.
    --print(confusion)
  end
  
  -- Clean temporary data to reduce the size of the network file.
  network:clearState()
  -- Save the network.
  torch.save('output.t7',network)


-------------------- 5. TEST NETWORK --------------------
testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
for i=1,1 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end
 --[[debug:]] 
-- TEST particular datapoint
--for fun, print the mean and standard-deviation of example-100
--print(classes[testset.label[100]])
--number = testset.data[100]
--print(number:mean(), number:std())
-- print(classes[testset.label[100]])
--[[
predicted = network:forward(testset.data[100])
-- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x 
print(predicted:exp())
for i=1,predicted:size(1) do
    print(classes[i], predicted[i])
end]]

--a) Accuracy
correct = 0
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = network:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/10000 .. ' % ')


--b) Class Performance
class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
class_count={0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1,10000 do
    local groundtruth = testset.label[i]
    class_count[groundtruth]=class_count[groundtruth]+1
    local prediction = network:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end

for i=1,#classes do
    print(classes[i], 100*class_performance[i]/class_count[i] .. ' %')
end



end --end of main()


main()
