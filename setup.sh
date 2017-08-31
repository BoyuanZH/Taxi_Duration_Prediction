#!/bin/bash
echo "This is a shell script for setting up a remote compute engine- debian"
sudo apt-get install git python python-setuptools
sudo easy_install pip
sudo apt-get install python-dev python-numpy python-scipy python-matplotlib python-pandas
sudo pip install --upgrade pandas sklearn
sudo pip install seaborn
sudo apt-get install g++ gfortran
sudo apt-get install libatlas-base-dev
sudo apt-get install make
echo "Python and packages needed for xgboost are installed"

# set up xgboost
git clone --recursive https://github.com/dmlc/xgboost.git
cd xgboost
make
cd python-package/
sudo python setup.py install
echo "Xgboost is installed."

# mkdir taxi
# gsutil cp -r gs://taxi_trip_duration/TaxiTripDurationPredict/ ~/taxi

