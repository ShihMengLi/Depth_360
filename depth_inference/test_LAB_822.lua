require 'nn'
require 'image'
require 'csvigo'
require 'hdf5'
require 'cudnn'
require 'cunn'
require 'cutorch'
require "trepl"
npy4th = require 'npy4th'

--Need Mean and Variance of the indoor scene to convert relative depth to absolute depth
--[[
function normalize_output_depth_with_NYU_mean_std( input )
    local std_of_NYU_training = 0.6148231626
    local mean_of_NYU_training = 2.8424594402
    
    local transformed_weifeng_z = input:clone()
    transformed_weifeng_z = transformed_weifeng_z - torch.mean(transformed_weifeng_z);
    transformed_weifeng_z = transformed_weifeng_z / torch.std(transformed_weifeng_z);
    transformed_weifeng_z = transformed_weifeng_z * std_of_NYU_training;
    transformed_weifeng_z = transformed_weifeng_z + mean_of_NYU_training;
    
    -- remove and replace the depth value that are negative
    if torch.sum(transformed_weifeng_z:lt(0)) > 0 then
        -- fill it with the minimum of the non-negative plus a eps so that it won't be 0
        transformed_weifeng_z[transformed_weifeng_z:lt(0)] = torch.min(transformed_weifeng_z[transformed_weifeng_z:gt(0)]) + 0.00001
    end

    return transformed_weifeng_z
end
]]--

cmd = torch.CmdLine()
cmd:text('Options')

cmd:option('-pretrained_model','','Absolute / relative path to the previous model file. Resume training from this file')
cmd:option('-image_list','','Input images set')
cmd:option('-output_path','','Output dir')
cmd_params = cmd:parse(arg)


-- Load the model
prev_model_file = cmd_params.pretrained_model
model = torch.load(prev_model_file)
model:evaluate()
print("Model file:", prev_model_file)


network_input_height = 240
network_input_width = 320
_batch_input_cpu = torch.Tensor(1,3,network_input_height,network_input_width)


image_list = csvigo.load({path = cmd_params.image_list, mode = "large"})
num_pictures = #image_list

-------------------------------
for i = 1, num_pictures do   

    -- read image, scale it to the input size
    local img = image.load(image_list[i][1])
    local img_original_height = img:size(2)
    local img_original_width = img:size(3)

    _batch_input_cpu[{1,{}}]:copy( image.scale(img,network_input_width ,network_input_height))


    -- forward
    local batch_output = model:forward(_batch_input_cpu:cuda());  
    cutorch.synchronize()
    local temp = batch_output
    if torch.type(batch_output) == 'table' then
        batch_output = batch_output[1]
    end
    save_img = batch_output
    save_img = save_img - torch.min(save_img)
    save_img = torch.div(save_img, torch.max(save_img))
    save_img = torch.reshape(save_img, 1, save_img:size(3), save_img:size(4))
    save_img = save_img:float() 
    save_img = image.scale(save_img, img_original_width, img_original_height)
    print(save_img:size())
    image.save(cmd_params.output_path  .. '/' .. tostring(i) .. '.jpg', save_img)
    save_img = torch.reshape(save_img, save_img:size(2), save_img:size(3))
    npy4th.savenpy(cmd_params.output_path .. '/' .. tostring(i) .. '.npy', save_img)
end
