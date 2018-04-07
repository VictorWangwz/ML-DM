# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from skimage.io import imread, imshow, imsave


# our code
from naive_bayes import NaiveBayes

from decision_stump import DecisionStump, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest
# from random_forest import RandomForest # TODO
from kmedians import Kmedians
from kmeans import Kmeans
# from kmedians import Kmedians # TODO
from quantize_image import ImageQuantizer
from sklearn.cluster import DBSCAN

def plot_2dclustering(X,y):
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.title('Cluster Plot')


def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == '1.2':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        groupnames = dataset["groupnames"]
        wordlist = dataset["wordlist"]

        # ADDED CODE HERE
        print(X)
        print(y)
        print(X_valid)
        print(y_valid)
        print(groupnames)

        print("3rd groupname is: %s" % groupnames[3])

        print(wordlist)

        print("50th word is: %s" % wordlist[50])
        print("Training example 500:")

        training_example_500 = X[500, :]

        var = np.where(training_example_500)

        for counn in var:
            print("Word at 500 isssss: %s" % wordlist[counn])

        count = 0
        for x in np.nditer(training_example_500):
            if x == 1:
                print("the index is: %d " % count)
            count += 1

        print(training_example_500)
        print("Words present in training example 500: ")
        print("Word at index 5: %s" % wordlist[5])
        print("Word at index 25: %s" % wordlist[25])
        print("Word at index 35: %s" % wordlist[35])
        print("Word at index 92: %s" % wordlist[92])

        print("Value corresponding to specific newsgroup: ")
        print(y[500])



    elif question == '1.3':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]

        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = RandomForestClassifier()
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Random Forest (sklearn) validation error: %.3f" % v_error)

        model = NaiveBayes(num_classes=4)
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (ours) validation error: %.3f" % v_error)

        model = BernoulliNB()
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (sklearn) validation error: %.3f" % v_error)

    elif question == '2':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)


        print("Our implementations:")
        print("  Decision tree info gain")
        evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))
        print("  Random tree info gain")
        evaluate_model(RandomTree(max_depth=np.inf))
        print("  Random forest info gain")
        evaluate_model(RandomForest(max_depth=np.inf, num_trees=50)) # TODO: implement this

        print("sklearn implementations")
        print("  Decision tree info gain")
        evaluate_model(DecisionTreeClassifier(criterion="entropy"))
        print("  Random forest info gain")
        evaluate_model(RandomForestClassifier(criterion="entropy"))
        print("  Random forest info gain, more trees")
        evaluate_model(RandomForestClassifier(criterion="entropy", n_estimators=50))


    elif question == '3':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        model.fit(X)
        plot_2dclustering(X, model.predict(X))

        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
       


    elif question == '3.1':
    	X = load_dataset('clusterData2.pkl')['X']
    	minError = np.inf
    	for i in range(51):
            model = Kmeans(k=4)
            model.fit(X)
            error = model.error(X)
            if error < minError:
                minError = error
                saveModel=model
            if i==50:
            	# print(minError)
            	print('min error is %f'%minError)
            	plot_2dclustering(X,saveModel.predict(X))
            	fname3 = "C:\\Users\\wangzhen\\Desktop\\cpsc340\\g5e0b_u7p1b_a2-master\\figs\\q3_3_kmeans.png"
            	plt.savefig(fname3)
            	print("\nFigure saved as '%s'" % fname3) 

                


        # print(minError)
        #       

    elif question == '3.2':
        X = load_dataset('clusterData2.pkl')['X']
        # model=Kmeans(k=1)
        # model.fit(X)
        # error=model.error(X)
        # print(error)

        minError = np.zeros(10)
        for k in range(10):
            minError[k] = np.inf
            larg_slope=0
            slope_k=1
            for i in range(50):
                model = Kmeans(k=k+1)
                model.fit(X)
                error = model.error(X)
                if error < minError[k]:
                	minError[k] = error
            print(-minError[k]+minError[k-1])
            if k!=0:
                if -minError[k]+minError[k-1]>larg_slope:
                	larg_slope= -minError[k]+minError[k-1]
                	

               

        x1 = [i + 1 for i in range(10)]
        y1 = minError

        plt.figure(1)
        plot1 = plt.plot(x1, y1, 'b', label='minError')
        plt.title('Minimum error found across 50 random initializations with different k (from 1 to 10)')
        plt.xlabel('k')
        plt.ylabel('Error')
        plt.xlim(1, 10)
        plt.legend(loc='upper right')
        plt.savefig("C:\\Users\\wangzhen\\Desktop\\cpsc340\\g5e0b_u7p1b_a2-master\\figs\\q3_3_kmeans_error.png")

    elif question == '3.3':
        # X = load_dataset('clusterData2.pkl')['X']
        # minError=np.inf
        # for i in range(51):
        #     model = Kmedians(k=4)
        #     model.fit(X)
        #     error = model.error(X)
        #     if error < minError:
        #         minError = error
        #         saveModel=model
        #     if i==50:
        #     	# print(minError)
        #     	print('min error is %f'%minError)
        #     	plot_2dclustering(X,saveModel.predict(X))
        #     	fname3 = "C:\\Users\\wangzhen\\Desktop\\cpsc340\\g5e0b_u7p1b_a2-master\\figs\\q3_3_kmedians.png"
        #     	plt.savefig(fname3)
        #     	print("\nFigure saved as '%s'" % fname3) 
        X = load_dataset('clusterData2.pkl')['X']
        # model=Kmeans(k=1)
        # model.fit(X)
        # error=model.error(X)
        # print(error)

        minError = np.zeros(10)
        for k in range(10):
            minError[k] = np.inf
            larg_slope=0
            slope_k=1
            for i in range(50):
                model = Kmedians(k=k+1)
                model.fit(X)
                error = model.error(X)
                if error < minError[k]:
                	minError[k] = error
            print(-minError[k]+minError[k-1])
            if k!=0:
                if -minError[k]+minError[k-1]>larg_slope:
                	larg_slope= -minError[k]+minError[k-1]
                	

               

        x1 = [i + 1 for i in range(10)]
        y1 = minError

        plt.figure(1)
        plot1 = plt.plot(x1, y1, 'b', label='minError')
        plt.title('Minimum error found across 50 random initializations with different k (from 1 to 10)')
        plt.xlabel('k')
        plt.ylabel('Error')
        plt.xlim(1, 10)
        plt.legend(loc='upper right')
        plt.savefig("C:\\Users\\wangzhen\\Desktop\\cpsc340\\g5e0b_u7p1b_a2-master\\figs\\q3_3_kmedians_error.png")


    elif question == '3.4':
        X = load_dataset('clusterData2.pkl')['X']
        for i in range(4,200):

        	model = DBSCAN(eps=i, min_samples=3)
        	y = model.fit_predict(X)

        	print("%f,Labels (-1 is unassigned):"%i, np.unique(model.labels_))

        # plot_2dclustering(X,y)
        # fname = os.path.join("..", "figs", "clusterdata_dbscan.png")
        # plt.savefig(fname)
        # print("\nFigure saved as '%s'" % fname)


    elif question == '4':
        img = imread(os.path.join("..", "data", "mandrill.jpg"))

        # part 1: implement quantize_image.py
        # part 2: use it on the doge
        for b in [1,2,4,6]:
            quantizer = ImageQuantizer(b)
            q_img = quantizer.quantize(img)
            print(quantizer.colours.shape)
            d_img = quantizer.dequantize(q_img)
            
            plt.figure()
            plt.imshow(d_img)
            fname = os.path.join("..", "figs", "b_{}_image.png".format(b))
            plt.savefig(fname)
            print("Figure saved as '%s'" % fname)

            plt.figure()
            plt.imshow(quantizer.colours[None] if b/2!=b//2 else np.reshape(quantizer.colours, (2**(b//2),2**(b//2),3)))
            plt.title("Colours learned")
            plt.xticks([])
            plt.yticks([])
            fname = os.path.join("..", "figs", "b_{}_colours.png".format(b))
            plt.savefig(fname)
            print("Figure saved as '%s'" % fname)


    else:
        print("Unknown question: %s" % question)
