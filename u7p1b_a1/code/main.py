# standard Python imports
import os
import argparse
import time
import pickle

import csv

# 3rd party libraries
import numpy as np                              # this comes with Anaconda
import matplotlib.pyplot as plt                 # this comes with Anaconda
import pandas as pd                             # this comes with Anaconda
from sklearn.tree import DecisionTreeClassifier # see http://scikit-learn.org/stable/install.html
from sklearn.neighbors import KNeighborsClassifier # same as above
from sklearn.cross_validation import train_test_split
# CPSC 340 code
import utils
from decision_stump import DecisionStump, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from knn import KNN, CNN
import simple_decision

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=["1.1", "2", "2.2", "2.3", "2.4", "3", "3.1", "3.2", "4.1", "4.2", "5"])

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1.1":
    	# Load the fluTrends dataset
        df = pd.read_csv(os.path.join('..','data','fluTrends.csv'))
        X = df.values
        names = df.columns.values
        #print(X)
        #print(names)

        ''' YOUR CODE HERE FOR Q1.1 '''
        #1
        a=np.zeros(2)
        x=np.array(X)
        print("The mean of all dataset is: %.4f"% np.mean(x))
        print("The maximum of all dataset is :%.4f"%np.max(x))
        print("The minimum of all dataset is :%.4f"%np.min(x))
        print("The median of all dataset is :%.4f"%np.median(x))
        print("The mode of all dataset is :%.4f"%utils.mode(x))
        #2
        print("The 5%% quantile across the dataset: %.4f"%np.percentile(x,5))
        print("The 25%% quantile across the dataset: %.4f"%np.percentile(x,25))
        print("The 50%% quantile across the dataset: %.4f"%np.percentile(x,50))
        print("The 75%% quantile across the dataset: %.4f"%np.percentile(x,75))
        print("The 95%% quantile across the dataset: %.4f"%np.percentile(x,95))
        #3
        columnMean=np.mean(x,axis=0)
        columnVar=np.var(x,axis=0)
        print("The regions with the highest mean:%s"%names[columnMean.tolist().index(np.max(columnMean))])
        print("The mean is:%.4f"%np.max(columnMean))
        print("The regions with the lowest mean:%s"%names[columnMean.tolist().index(np.min(columnMean))])
        print("The mean is:%.4f"%np.min(columnMean))
        print("The regions with the highest variance:%s"%names[columnVar.tolist().index(np.max(columnVar))])
        print("The variance is:%.4f"%np.max(columnVar))
        print("The regions with the lowest variance:%s"%names[columnVar.tolist().index(np.min(columnVar))])
        print("The variance is:%.4f"%np.min(columnVar))

        # x=sorted(x)
        # print(x)


        #pass
    
    elif question == "2":

        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate majority predictor model
        y_pred = np.zeros(y.size) + utils.mode(y)

        error = np.mean(y_pred != y)
        print("Mode predictor error: %.3f" % error)

        # 3. Evaluate decision stump
        model = DecisionStumpEquality()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y) 
        print("Decision Stump with equality rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == "2.2":
        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        # 3. Evaluate decision stump
        model = DecisionStump()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q2_2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == "2.3":
        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate decision tree
        model = DecisionTree(max_depth=2)
        model.fit(X, y)

        y_pred = model.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)

        # y_pred = simple_decision.predict(X)
        # error = np.mean(y_pred != y)
        # print("Simple Decision Error: %.4f" % error)


    
    elif question == "2.4":
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]
        print("n = %d" % X.shape[0])

        depths = np.arange(1,15) # depths to try

        t = time.time()
        my_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("Our decision tree took %f seconds" % (time.time()-t))
        
        plt.plot(depths, my_tree_errors, label="mine")
        
        t = time.time()
        sklearn_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("scikit-learn's decision tree took %f seconds" % (time.time()-t))

        
        plt.plot(depths, my_tree_errors, label="sklearn")
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q2_4_tree_errors.pdf")
        plt.savefig(fname)
        
        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)


    elif question == "3":
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]        
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "3.1":
        ''' YOUR CODE HERE FOR Q3.1 '''
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        #print("n = %d" % X.shape[0])

        depths = np.arange(1,15) # depths to try

        t = time.time()
        my_tree_errors = np.zeros(depths.size)
        my_tree_test_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(criterion='entropy', max_depth=i+1, random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            y_pred_test=model.predict(X_test)
            my_tree_errors[i] = np.mean(y_pred != y)
            my_tree_test_errors[i]=np.mean(y_pred_test!=y_test)
            print(my_tree_test_errors[i])
        #print("Our decision tree took %f seconds" % (time.time()-t))
        
        plt.plot(depths, my_tree_errors, label="train")
        plt.plot(depths, my_tree_test_errors, label="test")
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q3.1_train_test_errors.pdf")
        plt.savefig(fname)
        #pass

    elif question == "3.2":
        ''' YOUR CODE HERE FOR Q3.2 '''
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=0.5)
        depths = np.arange(1,15) # depths to try

        t = time.time()
        my_tree_errors = np.zeros(depths.size)
        my_tree_vali_errors = np.zeros(depths.size)
        my_tree_errors1 = np.zeros(depths.size)
        my_tree_vali_errors1 = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(criterion='entropy', max_depth=i+1, random_state=1)
            model.fit(X_vali, y_vali)
            y_pred = model.predict(X_train)
            y_pred_vali=model.predict(X_vali)
            my_tree_errors[i] = np.mean(y_pred != y_train)
            my_tree_vali_errors[i]=np.mean(y_pred_vali!=y_vali)
            print("Training error: %.3f" % my_tree_vali_errors[i])
            print("Validation error: %.3f" % my_tree_errors[i])
            model1 = DecisionTreeClassifier(criterion='entropy', max_depth=i+1, random_state=1)
            model1.fit(X_train, y_train)
            y_pred1 = model1.predict(X_train)
            y_pred_vali1=model1.predict(X_vali)
            my_tree_errors1[i] = np.mean(y_pred1 != y_train)
            my_tree_vali_errors1[i]=np.mean(y_pred_vali1!=y_vali)
            print("Training error1: %.3f" % my_tree_errors1[i])
            print("Validation error1: %.3f" % my_tree_vali_errors1[i])
        #print("Our decision tree took %f seconds" % (time.time()-t))
        
        plt.plot(depths, my_tree_errors1, label="train1")
        plt.plot(depths, my_tree_vali_errors1, label="vali1")
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q3.2_train_vali_errors1.pdf")
        plt.savefig(fname)

        plt.plot(depths, my_tree_errors, label="vali")
        plt.plot(depths, my_tree_vali_errors, label="train")
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q3.2_train_vali_errors.pdf")
        plt.savefig(fname)
       

    if question == '4.1':
        ''' YOUR CODE HERE FOR Q4.1 '''
        dataset=load_dataset("citiesSmall.pkl")
        X,y=dataset["X"],dataset["y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        # print(y_test)
        model=KNN(k=1)
        # model=KNeighborsClassifier(n_neighbors=1)
        model.fit(X_train,y_train)
        y_hat=model.predict(X_train)
        train_error= np.mean(y_hat != y_train)
        y_hat_test=model.predict(X_test)
        test_error= np.mean(y_hat_test != y_test)
        print(train_error,test_error)
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q4.1_decisionBoundarywith.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        

    if question == '4.2':
        ''' YOUR CODE HERE FOR Q4.2 '''
        dataset=load_dataset("citiesBig1.pkl")
        X,y=dataset["X"],dataset["y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
       
        # # print(y_test)
        # model_CNN=CNN(k=1)
        # #model_CNN=KNeighborsClassifier(n_neighbors=1)
        # model_CNN.fit(X_train,y_train) 
        # t = time.time()
        # y_hat=model_CNN.predict(X_train)
        # print("It takes %f time to make a prediction"% (time.time()-t))
        # train_error= np.mean(y_hat != y_train)
        # y_hat_test=model_CNN.predict(X_test)
        # test_error= np.mean(y_hat_test != y_test)
        # print(train_error,test_error)

        # utils.plotClassifier(model_CNN, X, y)
        # fname = os.path.join("..", "figs", "q4_2_decisionBoundary.pdf")
        # plt.savefig(fname)
        # print("\nFigure saved as '%s'" % fname)


        depths = np.arange(1,15) # depths to try

       
        # my_tree_errors = np.zeros(depths.size)
        # my_tree_test_errors = np.zeros(depths.size)
        # for i, max_depth in enumerate(depths):
        model = DecisionTreeClassifier(criterion='entropy', random_state=1)
        model.fit(X_train, y_train)
        t = time.time()
        y_pred = model.predict(X_train)
        print("It takes %f time to make a prediction"% (time.time()-t))
        y_pred_test=model.predict(X_test)
        my_tree_errors = np.mean(y_pred != y_train)
        my_tree_test_errors=np.mean(y_pred_test!=y_test)
        print(my_tree_test_errors)
        #print("Our decision tree took %f seconds" % (time.time()-t))
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q4.2_Decision_tree_Boundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


        
