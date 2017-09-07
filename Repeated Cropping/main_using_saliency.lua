require 'xlua'
require 'optim'
require 'cunn'
require 'nn'
require 'cudnn'
require 'image'
require 'loadcaffe'
cudnn.benchmark = true
cudnn.fastest = true
torch.setdefaulttensortype('torch.FloatTensor')

start_from_scratch = false

if start_from_scratch then
	prototxt = './full_conv.prototxt'
	binary = 'saliency_full_conv.caffemodel'
	model = loadcaffe.load(prototxt, binary)
	model:add(nn.ReLU())
	model:add(nn.View(-1):setNumInputDims(3))

	classifier = nn.ConcatTable()

	tl_classifier = nn.Sequential()
	tl_classifier:add(nn.Linear(128, 512))
	tl_classifier:add(nn.ReLU())
	tl_classifier:add(nn.Dropout(0.5))
	tl_classifier:add(nn.Linear(512, 4))

	br_classifier = nn.Sequential()
	br_classifier:add(nn.Linear(128, 512))
	br_classifier:add(nn.ReLU())
	br_classifier:add(nn.Dropout(0.5))
	br_classifier:add(nn.Linear(512, 4))

	classifier:add(tl_classifier)
	classifier:add(br_classifier)
	model:add(classifier)

	model:cuda()
	cudnn.convert(model, cudnn)
	evalCounter = 0
else
	epoch = 1
	-- model = torch.load('model/composition-epoch-' .. epoch-1 .. '.net')
	model = torch.load('model/composition-1st-stage(epoch16).net')
	evalCounter = (epoch-1) * 2880
end
lrs_w = {0.1, nil, nil, nil, 0.1, nil, nil, nil, 0.1, nil, 0.1, nil, 0.1, nil, nil, 0.1, nil, nil, 0.1, nil, nil, 0.1, nil, nil}
lrs_b = {0.1, nil, nil, nil, 0.1, nil, nil, nil, 0.1, nil, 0.1, nil, 0.1, nil, nil, 0.1, nil, nil, 0.1, nil, nil, 0.1, nil, nil}
param, gradParam = model:getParameters()
param_lr_m = model:clone()
param_lr = param_lr_m:getParameters()
for i = 1,nn.Sequential.size(model)-1 do
if lrs_w[i] ~= nil then
param_lr_m:get(i).weight:fill(lrs_w[i])
end
if lrs_b[i] ~= nil then
param_lr_m:get(i).bias:fill(lrs_b[i])
end
end

param_lr_m:get(25):get(1):get(1).weight:fill(1.0)
param_lr_m:get(25):get(1):get(1).bias:fill(1.0)
param_lr_m:get(25):get(1):get(4).weight:fill(1.0)
param_lr_m:get(25):get(1):get(4).bias:fill(1.0)
param_lr_m:get(25):get(2):get(1).weight:fill(1.0)
param_lr_m:get(25):get(2):get(1).bias:fill(1.0)
param_lr_m:get(25):get(2):get(4).weight:fill(1.0)
param_lr_m:get(25):get(2):get(4).bias:fill(1.0)

-- model = torch.load('model/srcnn_epoch_10000.net')
-- epoch = 10001
print('<attentionNet> using model:')
print(model)
tl_crit = nn.CrossEntropyCriterion()
br_crit = nn.CrossEntropyCriterion()
criterion = nn.ParallelCriterion():add(tl_crit):add(br_crit)
criterion:cuda()

param, gradParam = model:getParameters()


-- load train and test patch dataset
require 'hdf5'
print('Loading test data...')
testFile = hdf5.open('./data/composition_test.h5', 'r')
testData = testFile:all()
testFile:close()

-- print('Loading train data...')
-- trainFile = hdf5.open('./composition_train.h5', 'r')
-- trainData = trainFile:all()
-- trainFile:close()
collectgarbage()

-- set trainer
batchSize = 32
learningRate = 0.001
momentum = 0.9
dampening = 0
learningRateDecay = 0.0005
weightDecay = 0.0005
testLogger = optim.Logger('logs/composition_test.log')
trainLogger = optim.Logger('logs/composition_train.log')

classes = {'-', '\\', '|', '*'}
confusion_tl = optim.ConfusionMatrix(classes)
confusion_br = optim.ConfusionMatrix(classes)
epoch = epoch or 1

function train(trainlist)
	local time = sys.clock()
	print('<trainer> on training set:')
	print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
	print("<trainer> learning rate: " .. learningRate / (1 + evalCounter * learningRateDecay))
	sample_count = 0
	for f = 1,#trainlist do
		-- print("File ", f, "Done.")
		collectgarbage()
		trainFile = hdf5.open(trainlist[f], 'r')
		local dataset = trainFile:all()
		trainFile:close()

		data = dataset.data
		label = dataset.label
		dataset.size = data:size()[1]

		for t = 1,dataset.size,batchSize do
			--print(f, t, sample_count)
			-- prepare input batch
			local inlow = data:sub(t,math.min(t+batchSize-1, dataset.size))
			local inputs = torch.Tensor((#inlow)[1], 3, 448, 448)
			for k = 1,(#inlow)[1] do
				image.scale(inputs[k], inlow[k], 'bicubic')
			end
			inputs = inputs:cuda()
			local targets = label:sub(t,math.min(t+batchSize-1, dataset.size))+1
			targets = targets:cuda()
			local targets_tl = targets:narrow(4,1,1)
			local targets_br = targets:narrow(4,2,1)
			local targets_both = {targets_tl, targets_br}
			--targets = targets:cuda()
			collectgarbage()

			local feval = function(x)
				if x ~= parameters then
					param:copy(x)
				end
				gradParam:zero()

				local outputs = model:forward(inputs) 
				-- outputs = outputs:float()
				-- outputs[1] = outputs[1]:float()
				-- outputs[2] = outputs[2]:float()
				local f = criterion:forward(outputs, targets_both)
				local df_do = criterion:backward(outputs, targets_both)
				-- df_do[1] = df_do[1]:cuda()
				-- df_do[2] = df_do[2]:cuda()
				model:backward(inputs, df_do)
				-- gradParam:clamp(-1/learningRate, 1/learningRate)

				for i = 1,batchSize do
					confusion_tl:add(outputs[1][i]:view(-1), targets_both[1][i]:view(-1)[1])
					confusion_br:add(outputs[2][i]:view(-1), targets_both[2][i]:view(-1)[1])
				end
				return f, gradParam
			end

			-- perform SGD step:
			sgdState = sgdState or {
	 	        	learningRate = learningRate,
		        	momentum = momentum,
		        	learningRateDecay = learningRateDecay,
		        	dampening = dampening,
	            	learningRates = param_lr,
	            	weightDecay = weightDecay
		        }
		     state = state or {
		     	evalCounter = evalCounter
		 	}
			optim.sgd(feval, param, sgdState, state)
			xlua.progress((f-1)*(dataset.size) + t, dataset.size*(#trainlist))
			sample_count = sample_count + 1
		end
	end
	evalCounter = state.evalCounter
	time = sys.clock() - time
	time = time / sample_count
	print("\r<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')


	print(confusion_tl)
	print(confusion_br)
	-- trainLogger:add{['% mean class accuracy (train set, top left)'] = confusion_tl.totalValid * 100,
				   -- ['% mean class accuracy (train set, bottom right)'] = confusion_br.totalValid * 100}

     trainLogger:add{['tl (-)'] = confusion_tl.valids[1] * 100,
     				['tl (\\)'] = confusion_tl.valids[2] * 100,
     				['tl (|)'] = confusion_tl.valids[3] * 100,
     				['tl (*)'] = confusion_tl.valids[4] * 100,
     				['br (-)'] = confusion_br.valids[1] * 100,
     				['br (\\)'] = confusion_br.valids[2] * 100,
     				['br (|)'] = confusion_br.valids[3] * 100,
     				['br (*)'] = confusion_br.valids[4] * 100}

	confusion_tl:zero()
	confusion_br:zero()


	if epoch > 3 and epoch % 2 == 0 then
	local filename = 'model/composition-epoch-' .. tostring(epoch) .. '.net'
		os.execute('mkdir -p ' .. sys.dirname(filename))
		print('<trainer> saving network to '..filename)
		torch.save(filename, model:clearState())
	end
	-- next epoch
	epoch = epoch + 1
end

function test(dataset)
	data = dataset.data
	label = dataset.label
	dataset.size = data:size()[1]	
	local err = 0
	local time = sys.clock()
	print('<trainer> on testing Set:')
	for t = 1,dataset.size,batchSize do
		xlua.progress(t, dataset.size)
		-- prepare input batch
		local inlow = data:sub(t,math.min(t+batchSize-1, dataset.size))
		local inputs = torch.Tensor((#inlow)[1], 3, 448, 448)
		for k = 1,(#inlow)[1] do
			image.scale(inputs[k], inlow[k], 'bicubic')
		end
		inputs = inputs:cuda()
		local targets = label:sub(t,math.min(t+batchSize-1, dataset.size))+1
		local targets_tl = targets:narrow(4,1,1)
		local targets_br = targets:narrow(4,2,1)
		local targets_both = {targets_tl, targets_br}
		-- targets = targets:cuda()

		local outputs = model:forward(inputs)
		for i = 1,batchSize do
			confusion_tl:add(outputs[1][i]:view(-1), targets_both[1][i]:view(-1)[1])
			confusion_br:add(outputs[2][i]:view(-1), targets_both[2][i]:view(-1)[1])
		end
	end
	time = sys.clock() - time
	time = time / dataset.size
	print("\r<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')
	print(confusion_tl)
	print(confusion_br)
	-- testLogger:add{['% mean class accuracy (test set, top left)'] = confusion_tl.totalValid * 100,
				   -- ['% mean class accuracy (test set, bottom right)'] = confusion_br.totalValid * 100}
     testLogger:add{['tl (-)'] = confusion_tl.valids[1] * 100,
     				['tl (\\)'] = confusion_tl.valids[2] * 100,
     				['tl (|)'] = confusion_tl.valids[3] * 100,
     				['tl (*)'] = confusion_tl.valids[4] * 100,
     				['br (-)'] = confusion_br.valids[1] * 100,
     				['br (\\)'] = confusion_br.valids[2] * 100,
     				['br (|)'] = confusion_br.valids[3] * 100,
     				['br (*)'] = confusion_br.valids[4] * 100}
	confusion_tl:zero()
	confusion_br:zero()
end

function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do 
    lines[#lines + 1] = line
  end
  return lines
end

local train_file = 'train.txt'
local train_list = lines_from(train_file)

for i = 1,100 do
	model:training()
	train(train_list)
	trainLogger:style{['tl (-)'] = '-',
 				['tl (\\)'] = '-',
 				['tl (|)'] = '-',
 				['tl (*)'] = '-',
 				['br (-)'] = '-',
 				['br (\\)'] = '-',
 				['br (|)'] = '-',
 				['br (*)'] = '-'}
 	trainLogger:plot()
	collectgarbage()

	if i % 1 == 0 then
		model:evaluate()		
		test(testData)

    		testLogger:style{['tl (-)'] = '-',
     				['tl (\\)'] = '-',
     				['tl (|)'] = '-',
     				['tl (*)'] = '-',
     				['br (-)'] = '-',
     				['br (\\)'] = '-',
     				['br (|)'] = '-',
     				['br (*)'] = '-'}
	-- 	testLogger:style{['MSE of dataset (test set)'] = '-'}
		testLogger:plot()
	end
end
