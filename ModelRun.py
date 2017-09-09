from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime as dt
import seaborn as sns
import random

T0 = dt.datetime.now()

print("Loading Data...", end = "")
t0 = dt.datetime.now()
train = pd.read_csv(os.path.join("data", "train.csv"))
test = pd.read_csv(os.path.join("data", "test.csv"))
t1 = dt.datetime.now()
print("Complete loading. Loading cost {} seconds".format((t1-t0).seconds))

train["pickup_datetime"] = pd.to_datetime(train.pickup_datetime)
train["dropoff_datetime"] = pd.to_datetime(train.dropoff_datetime)
test["pickup_datetime"] = pd.to_datetime(test.pickup_datetime)

train["pickup_date"] = train.pickup_datetime.dt.date
test["pickup_date"] = test.pickup_datetime.dt.date
train["check_trip_duration"] = abs(train.pickup_datetime - train.dropoff_datetime).map(lambda x: x.total_seconds())


#### PCA on longitude and latitude
long_border = [-74.03, -73.75]
lat_border = [40.63, 40.85]
t0 = dt.datetime.now()
from sklearn.decomposition import PCA
coords = np.vstack((train[["pickup_longitude", "pickup_latitude"]].values,
                    train[["dropoff_longitude", "dropoff_latitude"]].values, 
                    test[["pickup_longitude", "pickup_latitude"]].values,
                    test[["dropoff_longitude", "dropoff_latitude"]].values))

pca = PCA().fit(coords)
train["pickup_pca0"] = pca.transform(train[["pickup_longitude", "pickup_latitude"]].values)[:, 0]
train["pickup_pca1"] = pca.transform(train[["pickup_longitude", "pickup_latitude"]].values)[:, 1]
train["dropoff_pca0"] = pca.transform(train[["dropoff_longitude", "dropoff_latitude"]].values)[:, 0]
train["dropoff_pca1"] = pca.transform(train[["dropoff_longitude", "dropoff_latitude"]].values)[:, 1]
test["pickup_pca0"] = pca.transform(test[["pickup_longitude", "pickup_latitude"]].values)[:, 0]
test["pickup_pca1"] = pca.transform(test[["pickup_longitude", "pickup_latitude"]].values)[:, 1]
test["dropoff_pca0"] = pca.transform(test[["dropoff_longitude", "dropoff_latitude"]].values)[:, 0]
test["dropoff_pca1"] = pca.transform(test[["dropoff_longitude", "dropoff_latitude"]].values)[:, 1]
t1 = dt.datetime.now()
print("PCA costs time {} second".format((t1-t0).seconds))


# add on new features
#############################################################################


# geographical features
def haversine(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine(lat1, lng1, lat1, lng2)
    b = haversine(lat1, lng1, lat2, lng1)
    return a + b

def bearing(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

train["haversine"] = haversine(train.pickup_latitude.values, train.pickup_longitude.values,
                               train.dropoff_latitude.values, train.dropoff_longitude.values)
train["manhattan_distance"] = manhattan_distance(train.pickup_latitude.values, train.pickup_longitude.values,
                                                 train.dropoff_latitude.values, train.dropoff_longitude.values)
train["bearing"] = bearing(train.pickup_latitude.values, train.pickup_longitude.values,
                           train.dropoff_latitude.values, train.dropoff_longitude.values)

test["haversine"] = haversine(test.pickup_latitude.values, test.pickup_longitude.values,
                               test.dropoff_latitude.values, test.dropoff_longitude.values)
test["manhattan_distance"] = manhattan_distance(test.pickup_latitude.values, test.pickup_longitude.values,
                                                 test.dropoff_latitude.values, test.dropoff_longitude.values)
test["bearing"] = bearing(test.pickup_latitude.values, test.pickup_longitude.values,
                           test.dropoff_latitude.values, test.dropoff_longitude.values)

## add in center latitude/longitude
train.loc[:, 'center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2
train.loc[:, 'center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2
test.loc[:, 'center_latitude'] = (test['pickup_latitude'].values + test['dropoff_latitude'].values) / 2
test.loc[:, 'center_longitude'] = (test['pickup_longitude'].values + test['dropoff_longitude'].values) / 2

### I don't trust this pca_manhattan, but why not?
train['pca_manhattan'] = np.abs(train['dropoff_pca1'] - train['pickup_pca1']) + np.abs(train['dropoff_pca0'] - train['pickup_pca0'])
test['pca_manhattan'] = np.abs(test['dropoff_pca1'] - test['pickup_pca1']) + np.abs(test['dropoff_pca0'] - test['pickup_pca0'])



# datetime features
train["pickup_weekday"] = train.pickup_datetime.dt.weekday
train["pickup_hour"] = train.pickup_datetime.dt.hour
train["pickup_week_of_year"] = train.pickup_datetime.dt.weekofyear
train['pickup_week_hour'] = train['pickup_weekday'] * 24 + train['pickup_hour']
train['pickup_dayofyear'] = train['pickup_datetime'].dt.dayofyear
test["pickup_weekday"] = test.pickup_datetime.dt.weekday
test["pickup_hour"] = test.pickup_datetime.dt.hour
test["pickup_week_of_year"] = test.pickup_datetime.dt.weekofyear
test['pickup_week_hour'] = test['pickup_weekday'] * 24 + test['pickup_hour']
test['pickup_dayofyear'] = test['pickup_datetime'].dt.dayofyear
     
########### clustering.
## Q: what is the best clustering approach for spacial data.
### standard kmeans takes a looooooooong time to fit. (487 total_seconds)
## minibatch kmeans is an aternative cost less time. (11 seconds for 100 batch size; 6 secs for 10000 batch size)
# Q: how to choose batch size?
t0 = dt.datetime.now()
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters = 100, batch_size = 10000).fit(np.random.permutation(coords[:500000]))
train["pickup_cluster"] = kmeans.predict(train[["pickup_longitude", "pickup_latitude"]].values)
train["dropoff_cluster"] = kmeans.predict(train[["dropoff_longitude", "dropoff_latitude"]].values)
test["pickup_cluster"] = kmeans.predict(test[["pickup_longitude", "pickup_latitude"]].values)
test["dropoff_cluster"] = kmeans.predict(test[["dropoff_longitude", "dropoff_latitude"]].values)
t1 = dt.datetime.now()
print("MiniBatchKMeans Clustering cost {} seconds".format((t1-t0).seconds))


####### Temporal data and geospatical data aggregation
#### average_speed_h_pickup_cluster_dropoff_cluster_pickup_hour
# first build average haversine speed on train
train["trip_speed_h"] = (1000 * train.haversine / train.trip_duration).round(2)  # unit is m/s
gby_cols_list = [[ "pickup_cluster", "dropoff_cluster"]]
for gby_cols in gby_cols_list:
    df_grouped_mean = train.groupby(gby_cols).mean()[["trip_speed_h"]]
    df_grouped_mean.columns = ["avg_speed_h_%s" % ("_".join(gby_cols))]
    train = pd.merge(train, df_grouped_mean, how = "left", left_on = gby_cols, right_index = True)
    test = pd.merge(test, df_grouped_mean, how = "left", left_on = gby_cols, right_index = True)
## there are alot 10k nan in test data

#_________________________________________________________________________
########################3 following aggregation feature come from beluga
group_freq = '60min'
df_all = pd.concat((train, test))[['id', 'pickup_datetime', 'pickup_cluster', 'dropoff_cluster']]
train.loc[:, 'pickup_datetime_group'] = train['pickup_datetime'].dt.round(group_freq)
test.loc[:, 'pickup_datetime_group'] = test['pickup_datetime'].dt.round(group_freq)

# Count trips over 60min
df_counts = df_all.set_index('pickup_datetime')[['id']].sort_index()
df_counts['count_60min'] = df_counts.isnull().rolling(group_freq).count()['id']
train = train.merge(df_counts, on='id', how='left')
test = test.merge(df_counts, on='id', how='left')

# Count how many trips are going to each cluster over time
dropoff_counts = df_all \
    .set_index('pickup_datetime') \
    .groupby([pd.TimeGrouper(group_freq), 'dropoff_cluster']) \
    .agg({'id': 'count'}) \
    .reset_index().set_index('pickup_datetime') \
    .groupby('dropoff_cluster').rolling('240min').mean() \
    .drop('dropoff_cluster', axis=1) \
    .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index() \
    .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'dropoff_cluster_count'})

train['dropoff_cluster_count'] = train[['pickup_datetime_group', 'dropoff_cluster']].merge(dropoff_counts, on=['pickup_datetime_group', 'dropoff_cluster'], how='left')['dropoff_cluster_count'].fillna(0)
test['dropoff_cluster_count'] = test[['pickup_datetime_group', 'dropoff_cluster']].merge(dropoff_counts, on=['pickup_datetime_group', 'dropoff_cluster'], how='left')['dropoff_cluster_count'].fillna(0)

# Count how many trips are going from each cluster over time
df_all = pd.concat((train, test))[['id', 'pickup_datetime', 'pickup_cluster', 'dropoff_cluster']]
pickup_counts = df_all \
    .set_index('pickup_datetime') \
    .groupby([pd.TimeGrouper(group_freq), 'pickup_cluster']) \
    .agg({'id': 'count'}) \
    .reset_index().set_index('pickup_datetime') \
    .groupby('pickup_cluster').rolling('240min').mean() \
    .drop('pickup_cluster', axis=1) \
    .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index() \
    .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'pickup_cluster_count'})

train['pickup_cluster_count'] = train[['pickup_datetime_group', 'pickup_cluster']].merge(pickup_counts, on=['pickup_datetime_group', 'pickup_cluster'], how='left')['pickup_cluster_count'].fillna(0)
test['pickup_cluster_count'] = test[['pickup_datetime_group', 'pickup_cluster']].merge(pickup_counts, on=['pickup_datetime_group', 'pickup_cluster'], how='left')['pickup_cluster_count'].fillna(0)
#_________________________________________________________________________




######### add OSRM Features (scr: https://www.kaggle.com/oscarleo/new-york-city-taxi-with-osrm)
t0 = dt.datetime.now()
fr1 = pd.read_csv(os.path.join('data', 'fastest_routes_train_part_1.csv'), usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps'])
fr2 = pd.read_csv(os.path.join('data', 'fastest_routes_train_part_2.csv'), usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
test_street_info = pd.read_csv(os.path.join('data', 'fastest_routes_test.csv'),
                               usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
train_street_info = pd.concat((fr1, fr2))
train = train.merge(train_street_info, how='left', on='id')
test = test.merge(test_street_info, how='left', on='id')
t1 = dt.datetime.now()
print("Loading OSRM Features cost {} seconds".format((t1-t0).seconds))

####check the features
features = list(train.columns)
non_features = ["id", "pickup_datetime", "dropoff_datetime", 
                "check_trip_duration", "trip_duration", "store_and_fwd_flag", 
                "pickup_date", "trip_speed_h", 'pickup_datetime_group']
features = [i for i in features if i not in non_features]
print('We have %d features.' % len(features))

## since we will estimate the rmsle, y = log(duration + 1)
y = np.log(train.trip_duration.values + 1)

#### modelling (is k fold a good alternative for model selection?)
from sklearn.model_selection import train_test_split
xtrain, xvalid, ytrain, yvalid = train_test_split(train[features].values, y, test_size = 0.2, random_state = 42)

import xgboost as xgb
## got a error befor in DMatrix, since one of the feature's dtype = object, this can not be used in transform into DMatrix in xgb
# so just don't include it in the model

#d = xgb.DMatrix(train)
dtrain = xgb.DMatrix(xtrain, label = ytrain)
dvalid = xgb.DMatrix(xvalid, label = yvalid)
dtest = xgb.DMatrix(test[features].values)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

##### citation, following code is from beluga.
# Try different parameters! My favorite is random search :)
t0 = dt.datetime.now()
xgb_params = {'min_child_weight': 10,
              'eta': 0.05, 
              'colsample_bytree': 0.8, 
              'max_depth': 15,
            'subsample': 0.8, 
            'lambda': 1., 
            'booster' : 'gbtree', 
            'silent': 1,
            'eval_metric': 'rmse', 
            'objective': 'reg:linear'}
xgb_model = xgb.train(xgb_params, dtrain, num_boost_round = 2000, 
                       evals=watchlist, early_stopping_rounds = 10, 
                       maximize=False, verbose_eval=10)
t1 = dt.datetime.now()
print("Training the model costs {} minutes".format((t1-t0).seconds//60))
print("Model's best score is {0:.5f}".format(xgb_model.best_score))

ytest =xgb_model.predict(dtest)
print("shape is okay") if len(ytest) == test.shape[0] else "oops"
test["trip_duration"] = np.exp(ytest) - 1

fit, axes = plt.subplots(1, 1, sharex = True, sharey = True)
sns.distplot(yvalid, ax = axes, label = "valid", color = "green")
sns.distplot(ytest, ax = axes, label = "test", color = "red")
axes.legend(loc = 0)

test[["id", "trip_duration"]].to_csv("xgb_submission.csv.gz", index = False, compression = "gzip")


## Plot the xgboost feature importance barplot.
###############################
#def plot_feature_importance(xgb_model, features, axes):
#    importance = pd.concat(\
#                           [pd.Series(dict(zip(['f%i' % i for i in range(len(features))],features))), 
#                           pd.Series(xgb_model.get_fscore())], axis = 1)\
#                           .sort_values(by=1)
#    importance.columns = ['feature', 'importance']
#    sns.barplot(data = importance, x = 'importance', y = 'feature', orient = 'h', ax = axes)
#    axes.set_title("feature importance")
#
#f, axes = plt.subplots(1, 1)
#plot_feature_importance(xgb_model, features, axes)








