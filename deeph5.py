import numpy as np
from keras.models import load_model
import sys
np.set_printoptions(precision = 16,suppress=True) # avoid printing in scientific notation

# this code wil extract neural networks from hdf5 files 
# and write a fortran scipt to implement them

# it takes in two arguments:
# 1. the hdf5 file containing the NN model
# 2. the .dat file containing your unscaled training data (for scaling), with each feature in a different column

# !! note it currently has a bug that values less than 1e-4 (these are very rare, <1 per NN) will print in scientific notation, 
# so you'll need to go into the file it produces and fix these manually.
# TODO generalise to any number of layers, any matrix dimensions
# T P Galligan, 2018

outfile = input('Enter desired output file name: ')


#----------Load the model--------------
model = load_model(sys.argv[1])

W0 = model.layers[0].get_weights()[0]
W1 = model.layers[1].get_weights()[0]
W2 = model.layers[2].get_weights()[0]
b0 = model.layers[0].get_weights()[1]
b1 = model.layers[1].get_weights()[1]
b2 = model.layers[2].get_weights()[1]

#------calculate mean and std dev------
feats = np.loadtxt(sys.argv[2])
n_feats = str(np.shape(feats)[1]) # number of features as a string
mean = np.mean(feats,axis=0)
std = np.std(feats,axis=0)
#--------------------------------------

with open(outfile, 'w') as f:
	f.write('program deepMetal\n'
			'! This program takes the deepMetal neural'
			'network and implements it as fortran90 code, for implementation'
			' in RAMSES.\n'
			'! It takes in a vector f containing'
			'input features and outputs a metallic heating rate\n'
			'! It assumes that all the features arrive as base-10 logarithms.\n'

			'implicit none \n \n'
			'real, dimension(1,9)   :: f ! must be a row vector \n'
			
			'real, dimension(1,'+n_feats+')   :: mean\n'
			'real, dimension(1,'+n_feats+')   :: std\n'
			
			'real, dimension('+n_feats+',20)  :: W0\n'
			'real, dimension(20,20) :: W1\n'
			'real, dimension(20,1)  :: W2 ! must be a column vector\n'
			
			'real, dimension(1,20)  :: b0 ! the biases must be row vectors\n'
			'real, dimension(1,20)  :: b1\n'
			'real, dimension(1,1)   :: b2\n'
			
			'real, dimension(10,20) :: M0\n'
			'real, dimension(21,20) :: M1\n'
			'real, dimension(21,1)  :: M2 ! must be a column vector\n'

			'real :: coolrate_m\n'

			'read (*,*) f(1,1),f(1,2),f(1,3),f(1,4),f(1,5),f(1,6),f(1,7),f(1,8),f(1,9)\n \n')

	f.write('mean = reshape((/&\n')
	for i in range(np.shape(mean)[0]):
		if i != np.shape(mean)[0]-1: # not last entry
			f.write(str(mean[i])+'d0,')
			if ((i+1) % 4 == 0):
				f.write('&\n')
		else: # last entry
			f.write(str(b0[i])+'d0 /), shape(mean))')
	f.write('\n \n')

	f.write('std = reshape((/&\n')
	for i in range(np.shape(std)[0]):
		if i != np.shape(std)[0]-1: # not last entry
			f.write(str(std[i])+'d0,')
			if ((i+1) % 4 == 0):
				f.write('&\n')
		else: # last entry
			f.write(str(std[i])+'d0 /), shape(std))')
	f.write('\n \n')

	f.write('b0 = reshape((/&\n')
	for i in range(np.shape(b0)[0]):
		if i != np.shape(b0)[0]-1: # not last entry
			f.write(str(b0[i])+'d0,')
			if ((i+1) % 4 == 0):
				f.write('&\n')
		else: # last entry
			f.write(str(b0[i])+'d0 /), shape(b0))')
	f.write('\n \n')

	f.write('b1 = reshape((/&\n')
	for i in range(np.shape(b1)[0]):
		if i != np.shape(b1)[0]-1: # not last entry
			f.write(str(b1[i])+'d0,')
			if ((i+1) % 4 == 0):
				f.write('&\n')
		else: # last entry
			f.write(str(b1[i])+'d0 /), shape(b1))')
	f.write('\n \n')
	
	f.write('b2 = reshape((/&\n')
	for i in range(np.shape(b2)[0]):
		if i != np.shape(b2)[0]-1: # not last entry
			f.write(str(b2[i])+'d0,')
			if ((i+1) % 4 == 0):
				f.write('&\n')
		else: # last entry
			f.write(str(b2[i])+'d0 /), shape(b2))')
	f.write('\n \n')

	f.write('W0 = reshape((/&\n')
	for i in range(np.shape(W0)[0]):
		for j in range(np.shape(W0)[1]):
				if (i+1 != np.shape(W0)[0]) or (j+1 != np.shape(W0)[1]): #not last entry
					f.write(str(W0[i,j])+'d0,')
					if (((np.shape(W0)[1]*i + j)+1) % 4 == 0):
						f.write('&\n')
				else: # last entry
					f.write(str(W0[i,j])+'d0/)&\n , (/'+str(np.shape(W0)[0]))
					f.write(','+str(np.shape(W0)[1])+'/), order = (/2,1/))')

	f.write('\n \n')
	
	f.write('W1 = reshape((/&\n')
	for i in range(np.shape(W1)[0]):
		for j in range(np.shape(W1)[1]):
				if (i+1 != np.shape(W1)[0]) or (j+1 != np.shape(W1)[1]): # not last entry
					f.write(str(W1[i,j])+'d0,')
					if (((np.shape(W1)[1]*i + j)+1) % 4 == 0):
						f.write('&\n')
				else: # last entry
					f.write(str(W1[i,j])+'d0/)&\n , (/'+str(np.shape(W1)[0]))
					f.write(','+str(np.shape(W1)[1])+'/), order = (/2,1/))')
	f.write('\n \n')
	
	f.write('W2 = reshape((/&\n')
	for i in range(np.shape(W2)[0]):
		for j in range(np.shape(W2)[1]):
				if (i+1 != np.shape(W2)[0]) or (j+1 != np.shape(W2)[1]): # not last entry
					f.write(str(W2[i,j])+'d0,')
					if (((np.shape(W2)[1]*i + j)+1) % 4 == 0):
						f.write('&\n')
				else: # last entry
					f.write(str(W2[i,j])+'d0/)&\n , (/'+str(np.shape(W2)[0]))
					f.write(','+str(np.shape(W2)[1])+'/), order = (/2,1/))')
	f.write('\n \n')

	f.write('f = (f-mean)/std ! scaling \n')
	f.write('M0(1,:) = b0(1,:)\n'
			'M0(2:,:) = W0\n'
			'M1(1,:) = b1(1,:)\n'
			'M1(2:,:) = W1\n'
			'M2(1,1) = b2(1,1)\n'
			'M2(2:,1) = W2(:,1)\n'
			'\n'
			'write (*,*) matmul(max(0.,matmul(max(0.,matmul(f,W0) + b0), W1) + b1), W2) + b2\n'
			'end program deepMetal')

