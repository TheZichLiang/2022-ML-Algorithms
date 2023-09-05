import numpy as np
from scipy.stats import norm
import re
import datetime
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from sklearn.metrics import mean_squared_error, mean_absolute_error
#from keras.callbacks import History

#Processing Step
myfiledata = "C:\\Users\\fzlce\\Downloads\\InternshipDataNeed\\2018-2021CombHurrReduceForMLDataWONAN-999.txt"
#load the data,get rid of the bar character, delete the last line, and split each string
data = np.loadtxt(myfiledata, delimiter=',', skiprows = 1, dtype=str)

#define input and outputs
#We want to get rid of N, E, and W and add negatives if necessary
data = np.array([item for item in data if "NaN" not in item])
for i in range(len(data)):
    data[:, 5][i] = re.sub('N', '', data[:, 5][i])
    if data[:,6][i].find('W') != -1:
        data[:, 6][i]= "-" + re.sub('W', '', data[:, 6][i])
    else:
        data[:, 6][i] = re.sub('E', '', data[:, 6][i])
INPUTS = np.delete(data, [0,1,2,7,8,9], 1)
OUTPUTS = np.array(data[:, 7:9])
print(INPUTS)
print(OUTPUTS)

Date = INPUTS[:,0]
fmt = '%Y%m%d'
datearr = []
for s in Date:
    dt = datetime.datetime.strptime(s, fmt)
    tt = dt.timetuple()
    datearr.append(tt.tm_yday)
print(datearr)
print(len(datearr))

Time = INPUTS[:,1]
print(Time)
Time = (Time.astype(int)/100).astype(int)

train_xo=np.zeros(INPUTS.shape, dtype=float)
train_xo[:,0] = np.array(datearr).astype(float)
train_xo[:,1] = np.array(Time).astype(float)
train_xo[:,2:5] = np.array(INPUTS[:,2:5]).astype(float)

idx=np.arange(train_xo.shape[0])
print(idx.shape)
np.random.seed(10)
np.random.shuffle(idx)
print("---", idx.shape)
train_xo = train_xo[idx,:]
train_xo = train_xo.T
train_yo = np.array(OUTPUTS.T).astype(float)
print(train_yo.shape)
train_yo = train_yo[:,idx]

tx_mean=np.mean(train_xo, axis=1)
tx_std = np.std(train_xo, axis=1)
train_x=(train_xo.T-tx_mean)/tx_std
ty_mean=np.mean(train_yo, axis=1)
ty_std = np.std(train_yo, axis=1)
train_y=(train_yo.T-ty_mean)/ty_std
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size = 0.2, random_state =0)
print(train_y.shape, train_x.shape, test_y.shape, test_x.shape)
print(ty_mean, ty_std)

LEARNING_RATE_BASE= 0.00002
LEARNING_RATE_DECAY = 0.99

# model architecture
model= Sequential()
model.add(Dense(100, input_dim=5))  #Create 2 hidden layers, 1 output layer (1st hidden layer)
#model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(200))      #2nd hidden layer
#model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(100))      #2nd hidden layer
#model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(2, activation='linear'))    #output layer
#opt=keras.optimizers.Adam(learning_rate=LEARNING_RATE_BASE, decay=LEARNING_RATE_DECAY)
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    LEARNING_RATE_BASE,
    decay_steps= 500,
    decay_rate=LEARNING_RATE_DECAY,
    staircase=True)
opt=keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(loss='mse', optimizer=opt, metrics=['mae'])  #1st dem mse, 2st dem mae, or any other function def
history=model.fit(train_x, train_y,    #training 1800 x 200 training data using batch sizes of 512
          batch_size=550,
#          shuffle=True,
#          steps_per_epoch=1,
          epochs=500,
          verbose=2,
#          validation_data=(test_x.T, test_y),
          validation_split=0.1)
          #callbacks = [history])
print(history.history['val_loss'])
aa=model.evaluate(test_x, test_y)
bb = model.predict(test_x)
avgdiff = np.mean(bb - test_y)
std = np.std(bb - test_y)
print(aa, avgdiff, std)
updtest_y = test_y * ty_std + ty_mean
test_x = test_x * tx_std + tx_mean
updbb = bb * ty_std + ty_mean
p_l = updbb - updtest_y
meandiff_y = np.mean(updbb - updtest_y, axis = 0)
std_real_y = np.std(updbb - updtest_y, axis = 0)
print(np.mean(updtest_y,axis=0), np.std(updtest_y,axis=0), np.min(updtest_y,axis=0), np.max(updtest_y,axis=0)) 
print(meandiff_y, std_real_y)

MSES, MSEP = round(mean_squared_error(updtest_y[:,0], updbb[:,0]), 2), round(mean_squared_error(updtest_y[:,1], updbb[:,1]),2)
MAES, MAEP =  round(mean_absolute_error(updtest_y[:,0], updbb[:,0]), 2), round(mean_absolute_error(updtest_y[:,1], updbb[:,1]), 2)
print("MSE: ", [MSES, MSEP])
print("MAE: ", [MAES, MAEP])

#Optimized loss: 0.4373, Validation loss: 0.9626 
#New Optimized Loss: 0.4787, Validation Loss: 1.0323 (3 years)
from matplotlib import pyplot as plt
x = range(1, 501)
y = history.history['val_loss']
plt.plot(x,y)
plt.title("2018-2021 DNN Validation Loss Over Epochs")
plt.xlabel("Val_Loss")
plt.ylabel("Epochs")
plt.show()

a, b = np.polyfit(updbb[:,0], updtest_y[:,0], 1)
plt.scatter(updbb[:,0], updtest_y[:,0], color = 'green')
plt.plot(updbb[:,0], a* updbb[:,0] + b, color = 'blue')
plt.plot(updbb[:,0], updbb[:,0], color = 'red', label = 'y=x')
plt.title("DNN TC 2018-2021 Maximum Wind Speed Comparison")
plt.text(10,75, 'Linear fit slope: ' + str(round(a,4)) + " intercept: " + str(round(b,4)))
plt.xlabel('Y')
plt.ylabel("Y-hat")
plt.show()

a, b = np.polyfit(updbb[:,1], updtest_y[:,1], 1)
plt.scatter(updbb[:,1], updtest_y[:,1], color = 'green')
plt.plot(updbb[:,1], a* updbb[:,1] + b, color = 'blue')
plt.plot(updbb[:,1], updbb[:,1], color = 'red', label = 'y=x')
plt.title("DNN TC 2018-2021 Minimum Pressure Comparison")
plt.text(930,1020, 'Linear fit slope: ' + str(round(a,4)) + " intercept: " + str(round(b,4)))
plt.xlabel('Y')
plt.ylabel("Y-hat")
plt.show()

p_l = updbb - updtest_y
print(p_l[:,0])
import matplotlib.pyplot as plt
p_lmean = np.mean(p_l, axis=0)
p_lstd = np.std(p_l, axis=0)
p_lmedian = np.median(p_l, axis=0)
RSDS = np.median(abs(p_l[:,0]- p_lmedian[0])) *1.4826
RSDP = np.median(abs(p_l[:,1]- p_lmedian[1])) *1.4826
print("Mean: ", p_lmean, "Std: ", p_lstd)
print("Median: ", p_lmean, "RSDS: ", RSDS, "RSDP: ", RSDP)

plt.figure(figsize=(6, 5), dpi=200)
domains = np.linspace(np.min(p_l[:,0]), np.max(p_l[:,0]))
plt.plot(domains, norm.pdf(domains, p_lmean[0], p_lstd[0]), label = r'$\mu,\sigma\ curve$')
plt.plot(domains, norm.pdf(domains, p_lmedian[0], RSDS), label = r'$\eta, RSD\ curve$')
plt.hist(p_l[:,0], edgecolor = 'black', bins=20, color='#50C878', alpha=0.7, rwidth=0.85, density = True)
textstr = '\n'.join((r'$\mathcal{STATISTICS}$', r'$\mu=%.2f$' % (p_lmean[0], ),
    r'$\sigma=%.3f$' % (p_lstd[0], ),
    r'$\eta=%.3f$' % (p_lmedian[0], ),
    r'$\mathit{RSD}=%.3f$' % (RSDS, ),
    'N = '+str(len(p_l))))
props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
plt.text(-40, 0.05, textstr, fontsize=10,
        verticalalignment='top', bbox=props)

plt.title("DNN TC 2018-2021 Maximum Wind Speed Residuals", weight = 'bold')
plt.xlabel('Predicted WS - References (m/s)', weight = 'bold')
plt.ylabel('Normalized % of Samples', weight = 'bold')
plt.legend(loc='upper left')
plt.show()

plt.figure(figsize=(6, 5), dpi=200)
domainp = np.linspace(np.min(p_l[:,1]), np.max(p_l[:,1]))
plt.plot(domainp, norm.pdf(domainp, p_lmean[1], p_lstd[1]), label = r'$\mu,\sigma\ curve$')
plt.plot(domainp, norm.pdf(domainp, p_lmedian[1], RSDP), label = r'$\eta, RSD\ curve$')
plt.hist(p_l[:,1], edgecolor = 'black', bins=20, color='#0504aa', alpha=0.7, rwidth=0.85, density = True)
textstr = '\n'.join((r'$\mathcal{STATISTICS}$', r'$\mu=%.2f$' % (p_lmean[1], ),
    r'$\sigma=%.3f$' % (p_lstd[1], ),
    r'$\eta=%.3f$' % (p_lmedian[1], ),
    r'$\mathit{RSD}=%.3f$' % (RSDP, ),
    'N = '+str(len(p_l))))
props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
plt.text(40, 0.03, textstr, fontsize=10,
        verticalalignment='top', bbox=props)

plt.title("DNN TC 2018-2021 Mimimum Pressure Residuals", weight = 'bold')
plt.xlabel('Predicted WS - References (hpa)', weight = 'bold')
plt.ylabel('Normalized % of Samples', weight = 'bold')
plt.legend(loc='upper right')
plt.show()

from mpl_toolkits.basemap import Basemap
fig=plt.figure(figsize=(12, 6) )
ax=fig.add_subplot(111)
plt.subplots_adjust(left=0.1, bottom=0, right=0.9, top=1, wspace=0, hspace=0)

# Miller projection:
m=Basemap(projection='cyl',lat_ts=10,llcrnrlon=-180, urcrnrlon=180,llcrnrlat=-90,urcrnrlat=90, resolution='c')

# convert the lat/lon values to x/y projections.
x, y = m(test_x[:,3],test_x[:,2])

# Add a coastline and axis values.
m.drawcoastlines()
m.fillcontinents()

btsize=np.zeros(p_l[:,0].shape[0])
btsize[:]=5

m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0], fontsize=16)
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1], fontsize=16)

# Add a colorbar and title, and then show the plot.
cs=m.scatter(x,y, c=p_l[:,0], s=btsize, cmap="rainbow", alpha=0.5)
# plt.cm.jet)
cbar=m.colorbar(cs, location='right', fraction=0.001, pad=0.12)
cbar.set_label("m/s", fontsize=16)
cbar.ax.tick_params(labelsize=16)

#cbar.set_clim(-10, 10)
cs.set_clim(-5, 15)
plt.title('Global DNN Residual Wind Speed Distribution Map', fontsize=20)
plt.show()

fig=plt.figure(figsize=(12, 6) )
ax=fig.add_subplot(111)
plt.subplots_adjust(left=0.1, bottom=0, right=0.9, top=1, wspace=0, hspace=0)

# Miller projection:
m=Basemap(projection='cyl',lat_ts=10,llcrnrlon=-180, urcrnrlon=180,llcrnrlat=-90,urcrnrlat=90, resolution='c')

# convert the lat/lon values to x/y projections.
x, y = m(test_x[:,3],test_x[:,2])

# Add a coastline and axis values.
m.drawcoastlines()
m.fillcontinents()

btsize=np.zeros(p_l[:,1].shape[0])
btsize[:]=5

m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0], fontsize=16)
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1], fontsize=16)

# Add a colorbar and title, and then show the plot.
cs=m.scatter(x,y, c=p_l[:,1], s=btsize, cmap="rainbow", alpha=0.5)
# plt.cm.jet)
cbar=m.colorbar(cs, location='right', fraction=0.001, pad=0.12)
cbar.set_label("m/s", fontsize=16)
cbar.ax.tick_params(labelsize=16)

#cbar.set_clim(-10, 10)
cs.set_clim(-20, 10)
plt.title('Global DNN Residual Wind Speed Distribution Map', fontsize=20)
plt.show()