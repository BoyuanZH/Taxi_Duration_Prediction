This repo consists of python scripts and model information for my first Kaggle competition called "New York Taxi Trip Duration Prediction", which I participated 2017 summer.

The repo is built in the purpose of maintaining codes and making deploying models on different machine relatively easy.

Contents:

1. `EDA.py`
    
     Scripts for Explotory Data Analysis and Feature Engineering.
2. `ModelRun.py`
     
     Main scripts for training XGBoost model and output prediciton csv file.
     
3. `Tuning.py`
     
     Scripts for Hyper-parameter tuning, use traditional grid search k-fold cross validation function by scikit-learn package, as well as nested looping grid search.
     
4. `Tuning_parallel.py`
    
    Crafted grid search function enabling parallel processing on multi-core machine.
    
5. `setup.sh`
    
    Scripts for setting up linux (Ubuntu) environment for XGBoost.