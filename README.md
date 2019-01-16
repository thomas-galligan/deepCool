# deepCool
This repository contains the source code for the training and deployment of deepCool, a neural network that allows fast and accurate calculations of cooling rates in irradiated astrophysical gases, allowing for non-homogeneous radiation fields. The code is highly flexible, and is suitable for adaptation to individual use cases (change in input features, etc.). 

NN.py contains the source code for training deepCool for estimation of cooling rates (deepCool), heating rates (deepHeat) and metal-line only cooling rates (deepMetal). The neural network is saved in .h5 format. At present it takes as input CLOUDY calculations from the cells of a preprocessed RAMSES-rt galaxy where each column corresponds to the target values and the features for each cell. This is easy to change to the individual user's needs. 

deeph5.py reads the h5 file and outputs Fortran code for implementation in RAMSES-rt (https://arxiv.org/abs/1304.7126). It is robust to changes in feature dimension, but is currently not robust to changes in neural network architecture (number of layers, etc.). 

formation_history.py contains code for determining star formation rates from a simulated galaxy. The impact of our improved calculations on star formation history is currently being investigated.
