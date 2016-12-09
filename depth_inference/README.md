## Indoor Depth Inference by Deng Jia's work in torch

- Need 
	- torch7
		- hdf5
		- csvigo
	- cudnn

- How to run 
	- List your RGB image files in an .csv (ex. image_set.csv)
	- Store RGB image files in 'data'
	```
	th depth_inference.lua -pretrained_model pretrained_model/Best_model_period1.t7 -image_list data/LAB822/image_set.csv  -output_path output/LAB822

	``` 
	- If you run it succesfully, you will see .jpg file and .npy file in the output dir.
