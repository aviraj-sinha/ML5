#https://www.kaggle.com/yingwurenjian/chicago-divvy-bicycle-sharing-data
import pandas as pd
import numpy as np
import random as rd
import seaborn as sns
import matplotlib.pyplot as plt

filename = "chicago-divvy-bicycle-sharing-data/data.csv"
n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
s = 1000 #desired sample size
skip = sorted(rd.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
bike_rides = pd.read_csv(filename, skiprows=skip)

bike_rides.head()
bike_rides.shape

bike_rides.tripduration.max()
labels = ["{0} - {1}".format(i, i + 4) for i in range(0, 60, 5)]
bike_rides['group_duration'] = pd.cut(bike_rides.tripduration, range(0, 65, 5), right=False, labels=labels)
bike_rides[['group_duration','tripduration']]


bike_rides['usertype'].max()


bike_rides['gender'].value_counts()
bike_rides['to_station_id'].value_counts()
bike_rides['gender'].value_counts()


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
sns.countplot(x="group_duration", data=bike_rides)
plt.show()


#remove trip duation and make as the y variable categorical label, coordingates, usetypes were all subscriber, times were already give, station name already has id,
bike_rides = bike_rides.drop(columns=['trip_id','year','tripduration','latitude_end', 'longitude_end', 'latitude_start', 'longitude_start', 'usertype', 'starttime','stoptime','from_station_name','to_station_name'])

list(bike_rides.columns)

# X = bike_rides.drop(columns=["group_duration"])
# y = bike_rides["group_duration"]
# n_samples, n_features = X.shape
# n_classes = len(np.unique(y))
# print("n_samples: {}".format(n_samples))
# print("n_features: {}".format(n_features))
# print("n_classes: {}".format(n_classes))
# print(np.unique(y))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

bike_rides[categorical_headers] = bike_rides[categorical_headers].astype(str)

df_train, df_test = train_test_split(bike_rides, test_size=0.2)



#,'from_station_id','to_station_id'
encoders = dict()
categorical_headers = ['month','week','day','hour','gender','events']

#dp is the total number of docs at each station

# scale the numeric, continuous variables
numeric_headers = ['temperature','dpcapacity_start','dpcapacity_end']

for col in numeric_headers:
    df_train[col] = df_train[col].astype(np.float)
    df_test[col] = df_test[col].astype(np.float)

    ss = StandardScaler()
    df_train[col] = ss.fit_transform(df_train[col].values.reshape(-1, 1))
    df_test[col] = ss.transform(df_test[col].values.reshape(-1, 1))




bike_rides.info()


for col in categorical_headers + ['group_duration']:
    df_train[col] = df_train[col].str.strip()
    df_test[col] = df_test[col].str.strip()

    # integer encoded variables
    encoders[col] = LabelEncoder() # save the encoder
    df_train[col+'_int'] = encoders[col].fit_transform(df_train[col])
    df_test[col+'_int'] = encoders[col].transform(df_test[col])


categorical_headers_ints = [x+'_int' for x in categorical_headers]

# we will forego one-hot encoding right now and instead just scale all inputs
#   this is just to get an example running in Keras (don't ever do this)
feature_columns = categorical_headers_ints+numeric_headers
X_train =  ss.fit_transform(df_train[feature_columns].values).astype(np.float32)
X_test =  ss.transform(df_test[feature_columns].values).astype(np.float32)

y_train = df_train['group_duration'].values.astype(np.str)
y_test = df_test['group_duration'].values.astype(np.str)

print(feature_columns)







from sklearn import metrics as mt
import keras

keras.__version__

from keras.models import Sequential
from keras.layers import Dense, Activation, Input
from keras.layers import Embedding, Flatten, Concatenate
from keras.models import Model

# combine the features into a single large matrix
X_train = df_train[feature_columns].values
X_test = df_test[feature_columns].values

# This returns a tensor
inputs = Input(shape=(X_train.shape[1],))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(units=10, activation='relu')(inputs)
predictions = Dense(1,activation='sigmoid')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)

model.compile(optimizer='sgd',
              loss='mean_squared_error',
              metrics=['accuracy'])

model.summary()













#dimesnionality and stanard dsclaring needed



#which features to cross




#from the features the algortihm  should estimate how long the rides should take
# since it is more important to classify them
# this will be used to put more bikes in that area to earn more money since the time takes longer making more money and do more maintaince on that area as awell
# since the bikes are a capital good they can always be taken away meaning it is better to classify the instance as high duration
# prioritizing percision would help improve confidence and be sure that place is properly maintained the ones that are not maintained can be manually viewed
# this is less expensive than focussing on recall because that would be more instances of spending more on underutilized ones.
#In general people want ot err on the sid of spending less money and would rather allow a few complaints to fix it.


# i would use k fold cross validation to avoid the problems of class imbalance. This would allow even testing. I will shuffle to prevent any one  fold from having too much of the same time periodself.
#this would be an accurate way of testing because the data of the previous year is always on hand adn trends would not change unless the area characterics of the bikestation changed in the long-termself.
