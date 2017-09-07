require 'cudnn'
require 'nn'
require 'image'
require 'lfs'
require 'cutorch'
cutorch.setDevice(1)

sofar = os.clock()

cmd = torch.CmdLine()
cmd:option('-net', 'model/composition-epoch-10.net', 'trained model')
cmd:option('-dir', 'userstudy_case', 'target directory')
opt = cmd:parse(arg or {})

print(opt.net)
print(opt.dir)
model = torch.load(opt.net) 
model:evaluate()
for file in lfs.dir(opt.dir) do
if lfs.attributes(opt.dir..'/'..file, "mode") == "file" then
img = image.load(opt.dir..'/'..file, 3, 'float')
target_img = img:clone()
channel = img:size()[1]
height = img:size()[2]
width = img:size()[3]
-- bbox format -> left, top, right, bottom
cur_left = 1
cur_right = width
cur_top = 1
cur_bottom = height
for i=1,1000 do
--print(cur_left, cur_top, cur_right, cur_bottom)
cur_width = cur_right-cur_left + 1
cur_height = cur_bottom-cur_top + 1
crop_width = math.ceil(cur_width/100)
crop_height = math.ceil(cur_height/100)
cur_img = img:narrow(2, cur_top, cur_height)
cur_img = cur_img:narrow(3, cur_left, cur_width)
input = image.scale(cur_img, '448x448', 'bicubic')
input = input:view(1,3,448,448)
input = input:cuda()
output = model:forward(input)
prob_tl, idx_tl = torch.max(output[1], 2)
prob_br, idx_br = torch.max(output[2], 2)
idx_tl = idx_tl[1][1]
idx_br = idx_br[1][1]
stop_here = true
if idx_tl == 1 then
--  print('-')
  cur_left = cur_left + crop_width
  stop_here = false
elseif idx_tl == 2 then
--  print('\\')
  cur_left = cur_left + crop_width
  cur_top = cur_top + crop_height
  stop_here = false
elseif idx_tl == 3 then
--  print('|')
  cur_top = cur_top + crop_height
  stop_here = false
else
--  print('o')
end

if idx_br == 1 then
--  print('-')
  cur_right = cur_right - crop_width
  stop_here = false
elseif idx_br == 2 then
--  print('\\')
  cur_right = cur_right - crop_width
  cur_bottom = cur_bottom - crop_height
  stop_here = false
elseif idx_br == 3 then
--  print('|')
  cur_bottom = cur_bottom - crop_height
  stop_here = false
else
--  print('o')
end
left_seg = target_img:narrow(3,cur_left,1):narrow(2,cur_top,cur_bottom-cur_top+1)
left_seg:fill(0)
right_seg = target_img:narrow(3,cur_right,1):narrow(2,cur_top,cur_bottom-cur_top+1)
right_seg:fill(0)

top_seg = target_img:narrow(2,cur_top,1):narrow(3,cur_left,cur_right-cur_left+1)
top_seg:fill(0)
bot_seg = target_img:narrow(2,cur_bottom,1):narrow(3,cur_left,cur_right-cur_left+1)
bot_seg:fill(0)
if cur_bottom <= cur_top or cur_right <= cur_left then
  break
end
if stop_here then
  break
end


end
left_seg:narrow(1,1,1):fill(1)
right_seg:narrow(1,1,1):fill(1)
top_seg:narrow(1,1,1):fill(1)
bot_seg:narrow(1,1,1):fill(1)
image.save('userstudy_result/'..file, cur_img)
image.save('userstudy_progress/'..file, target_img)
--print(file)
end
end

sof2 = os.clock()
print ("Job took "..sof2 - sofar)

print('done.')
--itorch.image({y_crop, y_low_crop, output})
--itorch.image(y_low)
