HOW TO RUN THE fvcnn.py CODE

Download the DTD dataset:
https://www.robots.ox.ac.uk/~vgg/data/dtd/

Open the archive and extract it in the same folder where the fvcnn.py code is.
This is because the code searches ins "./dtd/images" for all folders containing pics.

Open the code. I suggest to run it cell-by-cell:
 - Cell 1: (Line 1 - Line 353), defines the class and its methods. 
		- Line 51 - Line 253 is the tf definition of the vgg16 conv layers
		- All the other methods are for FV computation
		- Ignore the "forward_propagate" method: is outdated 

 - Cell 2: (Line 354 - 401) identifies the dataset, sets up variables and passes them through
	   the convolutional layers. You can specify how big each batch should be and how many 
	   batches of data you want to load with the two corresponding variables. 
	   BE MINDFUL: if you're not using the full dataset, the pictures are randomly sampled 
	   (without repetition) from the dataset. Otherwise they are loaded in an orderly fashion.
	   To run, just specify batch_size and batch_to_load. The total amount of images propagated
	   will be batch_size*batch_to_load. 
	   Descriptors are stored as tensors (of size [N_pics, 7, 7 , 512]) in the mat_descripts 
	   variable. They are also stacked together as a bidimensional matrix (of size
	   [N_pics*7*7, 512]) in the descripts variable. The original images are store in 
	   the feed_imgs variable.
 - Cell 3:(Line 402-407) performs GMM clustering. Just run it
 - Cell 4: (Line 408-end) computes FVs. 
