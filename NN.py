#------Preliminaries--------------
# This code creates the deepCool, deepMetal and deepHeat neural networks 
# H. Katz and T.P. Galligan, 2018

import numpy as np
seed = 7
np.random.seed(seed)
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

#load in the data. all_data.dat contains: coolrate, heatrate, metrate, hden, met, temp, rad(6 bins)
data = np.loadtxt('./all_data.dat')

cooling_rates = data[:,0]
heating_rates = data[:,1]
metal_rates = data[:,2]

# we will work entirely in log space
data[:,5:] = np.log10(data[:,5:]) # take log of temp and rad fields

# extract the features
features = data[:,3:]

#We are dealing with a log quantity so we replace 0 cooling rates with the min that isn't 0
cooling_rates[cooling_rates <= 0.0] = cooling_rates[cooling_rates > 0.0].min()
heating_rates[heating_rates <= 0.0] = heating_rates[heating_rates > 0.0].min()
metal_rates[metal_rates <= 0.0] = metal_rates[metal_rates > 0.0].min()

#take log of rates
cooling_rates = np.log10(cooling_rates)
heating_rates = np.log10(heating_rates)
metal_rates = np.log10(metal_rates)

#-----------deepCool----------------

# Making the test - train split
X_train, X_test, y_train, y_test = train_test_split(features,cooling_rates, test_size = 0.2, random_state=42)

# Set up the standard scaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

# create deepCool NN model
model = Sequential()
model.add(Dense(20, input_dim=9, kernel_initializer='normal', activation='relu'))
model.add(Dense(20, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')

checkpointer = ModelCheckpoint(filepath='./deepCool_ptype.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')

#cross validation
X_train_t, X_cv, y_train_t, y_cv = train_test_split(X_train_std,y_train, test_size = 0.1, random_state=42)

model.fit(X_train_t, y_train_t, epochs=500, shuffle=True, batch_size=10000,validation_data=(X_cv,y_cv),callbacks=[checkpointer])

#load the best deepCool model
model1 = load_model('./deepCool_ptype.h5')

#evaluate performance
y_pred=model1.predict(X_test_std)
y_pred = y_pred.reshape(len(y_pred))
frac_diff = np.abs((10.0**y_pred-10.0**y_test))/(10.0**y_test)
diff = np.abs(y_pred-y_test)

# make histogram
bins = np.linspace(1,9,30)
inds = np.digitize(X_test[:,1], bins)

p99 = []
p90 = []
p50 = []
temps = []

for i in range(len(bins)):
    if (inds==i).sum() > 0:
        p99.append(np.percentile(frac_diff[inds==i],99))
        p90.append(np.percentile(frac_diff[inds==i],90))
        p50.append(np.percentile(frac_diff[inds==i],50))
        temps.append(0.5*(bins[i]+bins[i+1]))













#---------deepMetal---------------

# Making the test - train split for metals
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(features,metal_rates, test_size = 0.2, random_state=42)

# Set up the standard scaler
scaler = StandardScaler()
scaler.fit(X_train_m)
X_train_std = scaler.transform(X_train_m)
X_test_std = scaler.transform(X_test_m)

# create NN model for metals
model = Sequential()
model.add(Dense(20, input_dim=9, kernel_initializer='normal', activation='relu'))
model.add(Dense(20, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')

checkpointer = ModelCheckpoint(filepath='./deepMetal_ptype.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')

X_train_t, X_cv, y_train_tm, y_cv_m = train_test_split(X_train_std,y_train_m, test_size = 0.1, random_state=42)

model.fit(X_train_t, y_train_tm, epochs=1000, shuffle=True, batch_size=10000,validation_data=(X_cv,y_cv_m),callbacks=[checkpointer])

model1 = load_model('./deepMetal_ptype.h5')

y_pred=model1.predict(X_test_std)
y_pred = y_pred.reshape(len(y_pred))
frac_diff = np.abs((10.0**y_pred-10.0**y_test_m))/(10.0**y_test_m)
diff = np.abs(y_pred-y_test_m)

bins = np.linspace(1,9,30)
inds = np.digitize(X_test_m[:,1], bins)

p99 = []
p90 = []
p50 = []
temps = []

for i in range(len(bins)):
    if (inds==i).sum() > 0:
        p99.append(np.percentile(frac_diff[inds==i],99))
        p90.append(np.percentile(frac_diff[inds==i],90))
        p50.append(np.percentile(frac_diff[inds==i],50))
        temps.append(0.5*(bins[i]+bins[i+1]))












#-----------deepHeat-------------

# Making the test - train split for heating rates
X_train_heat, X_test_heat, y_train_heat, y_test_heat = train_test_split(features,heating_rates, test_size = 0.2, random_state=42)

# Set up the standard scaler
scaler = StandardScaler()
scaler.fit(X_train_heat)
X_train_std_heat = scaler.transform(X_train_heat)
X_test_std_heat = scaler.transform(X_test_heat)


# create NN model
model = Sequential()
model.add(Dense(20, input_dim=9, kernel_initializer='normal', activation='relu'))
model.add(Dense(20, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')

checkpointer = ModelCheckpoint(filepath='./deepHeat_ptype.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')

#cross validation
X_train_th, X_cvh, y_train_th, y_cvh = train_test_split(X_train_std_heat,y_train_heat, test_size = 0.1, random_state=42)

model.fit(X_train_th, y_train_th, epochs=1000, shuffle=True, batch_size=10000,validation_data=(X_cvh,y_cvh),callbacks=[checkpointer])

model1 = load_model('./deepHeat_ptype.h5')

y_pred=model1.predict(X_test_std_heat)
y_pred = y_pred.reshape(len(y_pred))
frac_diff = np.abs((10.0**y_pred-10.0**y_test_heat))/(10.0**y_test_heat)
diff = np.abs(y_pred-y_test_heat)


bins = np.linspace(1,9,30)
inds = np.digitize(X_test_heat[:,1], bins)

p99 = []
p90 = []
p50 = []
temps = []

for i in range(len(bins)):
    if (inds==i).sum() > 0:
        p99.append(np.percentile(frac_diff[inds==i],99))
        p90.append(np.percentile(frac_diff[inds==i],90))
        p50.append(np.percentile(frac_diff[inds==i],50))
        temps.append(0.5*(bins[i]+bins[i+1]))










