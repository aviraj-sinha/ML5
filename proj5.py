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
print(bike_rides.shape)
#remove trip duation and make as the y variable categorical label, coordingates, usetypes were all subscriber, times were already give, station name already has id,
bike_rides = bike_rides.drop(columns=['trip_id','year','latitude_end', 'longitude_end', 'latitude_start', 'longitude_start', 'usertype', 'starttime','stoptime','from_station_name','to_station_name'])
bike_rides.head()

bike_rides.tripduration.max()
labels = ["{0} - {1}".format(i, i + 4) for i in range(0, 60, 5)]


bike_rides['tripduration'].max()


bike_rides['gender'].value_counts()
bike_rides['to_station_id'].value_counts()
bike_rides['gender'].value_counts()


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
sns.distplot(a=bike_rides["tripduration"])
plt.show()

list(bike_rides.columns)
bike_rides.loc[bike_rides['tripduration'] < 10,'group_duration'] = "LOW"
bike_rides.loc[bike_rides['tripduration'] > 10,'group_duration'] = "HIGH"
bike_rides.head()



















from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# bike_rides[categorical_headers] = bike_rides[categorical_headers].astype(str)
categorical_headers = ['month','week','day','hour','gender','events']
bike_rides[categorical_headers] = bike_rides[categorical_headers].astype(str)

df_train, df_test = train_test_split(bike_rides, test_size=0.2)
df_train.dropna(inplace=True)
df_train.reset_index()

df_test.dropna(inplace=True)
df_test.reset_index()
print(df_test.group_duration.value_counts())

#,'from_station_id','to_station_id'
encoders = dict()

#dp is the total number of docs at each station

for col in categorical_headers + ['group_duration'] :
    df_train[col] = df_train[col].str.strip()
    df_test[col] = df_test[col].str.strip()

    if col=="group_duration":
        tmp = LabelEncoder()
        df_train[col] = tmp.fit_transform(df_train[col])
        df_test[col] = tmp.transform(df_test[col])
    else:
        # integer encoded variables
        encoders[col] = LabelEncoder()
        df_train[col+'_int'] = encoders[col].fit_transform(df_train[col])
        df_test[col+'_int'] = encoders[col].transform(df_test[col])


# scale the numeric, continuous variables
numeric_headers = ['temperature','dpcapacity_start','dpcapacity_end']

for col in numeric_headers :
    df_train[col] = df_train[col].astype(np.float)
    df_test[col] = df_test[col].astype(np.float)

    ss = StandardScaler()
    df_train[col] = ss.fit_transform(df_train[col].values.reshape(-1, 1))
    df_test[col] = ss.transform(df_test[col].values.reshape(-1, 1))

df_test.head()
categorical_headers_ints = [x+'_int' for x in categorical_headers]

# we will forego one-hot encoding right now and instead just scale all inputs
#   this is just to get an example running in Keras (don't ever do this)
feature_columns = categorical_headers_ints+numeric_headers
X_train =  ss.fit_transform(df_train[feature_columns].values).astype(np.float32)
X_test =  ss.transform(df_test[feature_columns].values).astype(np.float32)

y_train = df_train['group_duration'].values.astype(np.int)
y_test = df_test['group_duration'].values.astype(np.int)

print(feature_columns)

df_train[feature_columns].head()
df_train["group_duration"].head()



















from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# bike_rides[categorical_headers] = bike_rides[categorical_headers].astype(str)
categorical_headers = ['month','week','day','hour','gender','events']
bike_rides[categorical_headers] = bike_rides[categorical_headers].astype(str)

df_train, df_test = train_test_split(bike_rides, test_size=0.2)
df_train.dropna(inplace=True)
df_train.reset_index()

df_test.dropna(inplace=True)
df_test.reset_index()
print(df_test.group_duration.value_counts())

#,'from_station_id','to_station_id'
encoders = dict()

#dp is the total number of docs at each station

for col in categorical_headers + ['group_duration'] :
    df_train[col] = df_train[col].str.strip()
    df_test[col] = df_test[col].str.strip()

    if col=="group_duration":
        tmp = LabelEncoder()
        df_train[col] = tmp.fit_transform(df_train[col])
        df_test[col] = tmp.transform(df_test[col])
    else:
        # integer encoded variables
        encoders[col] = LabelEncoder()
        df_train[col+'_int'] = encoders[col].fit_transform(df_train[col])
        df_test[col+'_int'] = encoders[col].transform(df_test[col])


# scale the numeric, continuous variables
numeric_headers = ['temperature','dpcapacity_start','dpcapacity_end']

for col in numeric_headers :
    df_train[col] = df_train[col].astype(np.float)
    df_test[col] = df_test[col].astype(np.float)

    ss = StandardScaler()
    df_train[col] = ss.fit_transform(df_train[col].values.reshape(-1, 1))
    df_test[col] = ss.transform(df_test[col].values.reshape(-1, 1))

df_test.head()





from sklearn.preprocessing import OneHotEncoder
categorical_headers_ints = [x+'_int' for x in categorical_headers]

# we will forego one-hot encoding right now and instead just scale all inputs
#   this is just to get an example running in Keras (don't ever do this)
feature_columns = categorical_headers_ints+numeric_headers
X_train =  ss.fit_transform(df_train[feature_columns].values).astype(np.float32)
X_test =  ss.transform(df_test[feature_columns].values).astype(np.float32)

y_train = df_train['group_duration'].values.astype(np.int)
y_test = df_test['group_duration'].values.astype(np.int)


ohe = OneHotEncoder()
X_train_ohe = ohe.fit_transform(df_train[categorical_headers_ints].values)
X_test_ohe = ohe.transform(df_test[categorical_headers_ints].values)

# and save off the numeric features
X_train_num =  df_train[numeric_headers].values
X_test_num = df_test[numeric_headers].values

print(feature_columns)


















from sklearn import metrics as mt
import keras
from keras import regularizers
keras.__version__

from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Dropout
from keras.layers import Embedding, Flatten, Concatenate
from keras.layers import concatenate
from keras.models import Model




from sklearn.preprocessing import OneHotEncoder

cross_columns = [['month','week','day'],['events','hour']]


#'workclass','education','marital_status','occupation','relationship','race','sex','country'

# we need to create separate lists for each branch
embed_branches = []
X_ints_train = []
X_ints_test = []
all_inputs = []
all_wide_branch_outputs = []

for cols in cross_columns:
    # encode crossed columns as ints for the embedding
    enc = LabelEncoder()

    # create crossed labels
    X_crossed_train = df_train[cols].apply(lambda x: '_'.join(x), axis=1)
    X_crossed_test = df_test[cols].apply(lambda x: '_'.join(x), axis=1)

    enc.fit(np.hstack((X_crossed_train.values,  X_crossed_test.values)))
    X_crossed_train = enc.transform(X_crossed_train)
    X_crossed_test = enc.transform(X_crossed_test)
    X_ints_train.append( X_crossed_train )
    X_ints_test.append( X_crossed_test )

    # get the number of categories
    N = max(X_ints_train[-1]+1) # same as the max(df_train[col])

    # create embedding branch from the number of categories
    inputs = Input(shape=(1,),dtype='int32', name = '_'.join(cols))
    all_inputs.append(inputs)
    x = Embedding(input_dim=N,
                  output_dim=int(np.sqrt(N)),
                  input_length=1, name = '_'.join(cols)+'_embed')(inputs)
    x = Flatten()(x)
    all_wide_branch_outputs.append(x)

# merge the branches together
wide_branch = concatenate(all_wide_branch_outputs, name='wide_concat')
wide_branch = Dense(units=1,activation='sigmoid',name='wide_combined')(wide_branch)

# reset this input branch
all_deep_branch_outputs = []
# add in the embeddings
for col in categorical_headers_ints:
    # encode as ints for the embedding
    X_ints_train.append( df_train[col].values )
    X_ints_test.append( df_test[col].values )

    # get the number of categories
    N = max(X_ints_train[-1]+1) # same as the max(df_train[col])

    # create embedding branch from the number of categories
    inputs = Input(shape=(1,),dtype='int32', name=col)
    all_inputs.append(inputs)
    x = Embedding(input_dim=N,
                  output_dim=int(np.sqrt(N)),
                  input_length=1, name=col+'_embed')(inputs)
    x = Flatten()(x)
    all_deep_branch_outputs.append(x)

# also get a dense branch of the numeric features
all_inputs.append(Input(shape=(X_train_num.shape[1],),
                        sparse=False,
                        name='numeric_data'))

x = Dense(units=20, activation='relu',name='numeric_1')(all_inputs[-1])
all_deep_branch_outputs.append( x )

# merge the deep branches together
deep_branch = concatenate(all_deep_branch_outputs,name='concat_embeds')
deep_branch = Dense(units=50,kernel_regularizer=regularizers.l2(0.01), activation='relu', name='deep1')(deep_branch)
deep_branch = Dropout(0.5)(deep_branch)
deep_branch = Dense(units=25,kernel_regularizer=regularizers.l2(0.01), activation='relu', name='deep2')(deep_branch)
deep_branch = Dropout(0.5)(deep_branch)
deep_branch = Dense(units=10,kernel_regularizer=regularizers.l2(0.01), activation='relu', name='deep3')(deep_branch)
deep_branch = Dropout(0.5)(deep_branch)

final_branch = concatenate([wide_branch, deep_branch],name='concat_deep_wide')
final_branch = Dense(units=1,activation='sigmoid',name='combined')(final_branch)

model = Model(inputs=all_inputs, outputs=final_branch)






from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# you will need to install pydot properly on your machine to get this running
SVG(model_to_dot(model).create(prog='dot', format='svg'))


model.compile(optimizer='adagrad',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# lets also add the history variable to see how we are doing
# and lets add a validation set to keep track of our progress
history = model.fit(X_ints_train+ [X_train_num],
                    y_train,
                    epochs=10,
                    batch_size=32,
                    verbose=1,
                    validation_data = (X_ints_test + [X_test_num], y_test))

yhat = np.round(model.predict(X_ints_test + [X_test_num]))
print(mt.confusion_matrix(y_test,yhat), mt.accuracy_score(y_test,yhat))

from matplotlib import pyplot as plt

%matplotlib inline

plt.figure(figsize=(10,4))
plt.subplot(2,2,1)
plt.plot(history.history['acc'])

plt.ylabel('Accuracy %')
plt.title('Training')
plt.subplot(2,2,2)
plt.plot(history.history['val_acc'])
plt.title('Validation')

plt.subplot(2,2,3)
plt.plot(history.history['loss'])
plt.ylabel('MSE Training Loss')
plt.xlabel('epochs')

plt.subplot(2,2,4)
plt.plot(history.history['val_loss'])
plt.xlabel('epochs')






from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold


new_df = pd.concat([df_train,df_test], axis=0)

new_df.dropna(inplace=True)
new_df.reset_index(inplace=True)

X = new_df.drop(columns=["group_duration"])
Y = new_df["group_duration"]

seed=7
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
    # create model
    # Fit the model
    X_ints_train = []
    X_ints_test = []
    all_inputs = []
    all_wide_branch_outputs = []
    embed_branches = []
    train_df = new_df.iloc[train,:]
    test_df = new_df.iloc[test,:]
    y_train = train_df['group_duration'].values.astype(np.int)
    y_test = test_df['group_duration'].values.astype(np.int)
    X_train_num =  train_df[numeric_headers].values
    X_test_num = test_df[numeric_headers].values


    for cols in cross_columns:
        # encode crossed columns as ints for the embedding
        enc = LabelEncoder()

        # create crossed labels
        X_crossed_train = train_df[cols].apply(lambda x: '_'.join(x), axis=1)
        X_crossed_test = test_df[cols].apply(lambda x: '_'.join(x), axis=1)

        enc.fit(np.hstack((X_crossed_train.values,  X_crossed_test.values)))
        X_crossed_train = enc.transform(X_crossed_train)
        X_crossed_test = enc.transform(X_crossed_test)
        X_ints_train.append( X_crossed_train )
        X_ints_test.append( X_crossed_test )

        # get the number of categories
        N = max(X_ints_train[-1]+1) # same as the max(df_train[col])

        # create embedding branch from the number of categories
        inputs = Input(shape=(1,),dtype='int32', name = '_'.join(cols))
        all_inputs.append(inputs)
        x = Embedding(input_dim=N,
                      output_dim=int(np.sqrt(N)),
                      input_length=1, name = '_'.join(cols)+'_embed')(inputs)
        x = Flatten()(x)
        all_wide_branch_outputs.append(x)

    # merge the branches together
    wide_branch = concatenate(all_wide_branch_outputs, name='wide_concat')
    wide_branch = Dense(units=1,activation='sigmoid',name='wide_combined')(wide_branch)

    # reset this input branch
    all_deep_branch_outputs = []
    # add in the embeddings
    for col in categorical_headers_ints:
        # encode as ints for the embedding
        X_ints_train.append( train_df[col].values )
        X_ints_test.append( test_df[col].values )

        # get the number of categories
        N = max(X_ints_train[-1]+1) # same as the max(df_train[col])

        # create embedding branch from the number of categories
        inputs = Input(shape=(1,),dtype='int32', name=col)
        all_inputs.append(inputs)
        x = Embedding(input_dim=N,
                      output_dim=int(np.sqrt(N)),
                      input_length=1, name=col+'_embed')(inputs)
        x = Flatten()(x)
        all_deep_branch_outputs.append(x)

    # also get a dense branch of the numeric features
    all_inputs.append(Input(shape=(X_train_num.shape[1],),
                sparse=False,
                name='numeric_data'))

    x = Dense(units=20, activation='relu',name='numeric_1')(all_inputs[-1])
    all_deep_branch_outputs.append( x )

    # merge the deep branches together
    deep_branch = concatenate(all_deep_branch_outputs,name='concat_embeds')
    deep_branch = Dense(units=50,kernel_regularizer=regularizers.l2(0.01), activation='relu', name='deep1')(deep_branch)
    deep_branch = Dropout(0.5)(deep_branch)
    deep_branch = Dense(units=25,kernel_regularizer=regularizers.l2(0.01), activation='relu', name='deep2')(deep_branch)
    deep_branch = Dropout(0.5)(deep_branch)
    deep_branch = Dense(units=10,kernel_regularizer=regularizers.l2(0.01), activation='relu', name='deep3')(deep_branch)
    deep_branch = Dropout(0.5)(deep_branch)

    final_branch = concatenate([wide_branch, deep_branch],name='concat_deep_wide')
    final_branch = Dense(units=1,activation='sigmoid',name='combined')(final_branch)

    model = Model(inputs=all_inputs, outputs=final_branch)






    from IPython.display import SVG
    from keras.utils.vis_utils import model_to_dot

    # you will need to install pydot properly on your machine to get this running
    SVG(model_to_dot(model).create(prog='dot', format='svg'))


    model.compile(optimizer='adagrad',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # lets also add the history variable to see how we are doing
    # and lets add a validation set to keep track of our progress
    history = model.fit(X_ints_train+ [X_train_num],
                        y_train,
                        epochs=10,
                        batch_size=32,
                        verbose=1,
                        validation_data = (X_ints_test + [X_test_num], y_test))

    yhat = np.round(model.predict(X_ints_test + [X_test_num]))


    cvscores.append(mt.accuracy_score(y_test,yhat))

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))





#grid search
import warnings
warnings.filterwarnings('ignore')

def score_func(y, y_pred, **kwargs):
    return recall_score(mapped_attack(y),y_pred, average="macro")

my_scorer = make_scorer(score_func)

vals = { 'n_hidden':10,
         'C':1e-2, 'epochs':200, 'eta':0.001,
         'alpha':0.001, 'decrease_const':1e-6, 'minibatches':10,
         'shuffle':True,'random_state':1, 'layers':4, 'cost':'entr', 'phi': 'lin'}

pipe_nn = Pipeline([('scaling', StandardScaler()),
                    ('pca', PCA(n_components=5)),
                    ('clf', MultilayerPerceptron(**vals))])
param_grid = {
    'clf__C':[1e-2],
    'clf__cost':["entr","quad"],
    'clf__layers':[2,3,4],
    'clf__phi':["lin", "sig","rilu","silu"],
    'clf__epochs':[400]
    }

cv = KFold(n_splits=5)

print(pipe_nn.get_params().keys())

gs = GridSearchCV(pipe_nn, param_grid,scoring=my_scorer, cv=cv, verbose=1)

best_model = gs.fit(X_train, y_train.values)
















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
