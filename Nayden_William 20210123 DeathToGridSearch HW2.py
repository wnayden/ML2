# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 10:50:50 2021

@author: WilliamNayden
"""
# adapt this code below to run your analysis
# 1. Write a function to take a list or dictionary of clfs and hypers(i.e. use logistic regression), each with 3 different sets of hyper parameters for each
# 2. Expand to include larger number of classifiers and hyperparameter settings
# 3. Find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function

#%%
import numpy as np
from sklearn.metrics import accuracy_score # other metrics too pls!
from sklearn.ensemble import RandomForestClassifier # more!
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import random
import itertools
from sklearn import preprocessing
from sklearn import datasets
import sys

#%%
def cartesian(params):
    """
    Creating a dictionary with Cartesian product of selected parameters
    based on https://docs.python.org/3/library/itertools.html

    Parameters
    ----------
    params : Dictionary
        All parameters in a selected model.

    Yields
    ------
    Dictionary
        Used later in grid search.

    """
    keys = params.keys()
    vals = params.values()
    for rowVals in itertools.product(*vals):
        yield dict(zip(keys, rowVals))
#%%       
def gridSearch(clfs,data):
    """
    Main function that runs the grid search across multiple classifiers and multiple parameters. 
    Results are saved in a dictionary with totals by model, parameters set, and details about the CV outputs
    
    Parameters
    ----------
    clfs : Dictionary
        All the models and their hyperparameters in a dictionary
    data : Tuple
        Your keys, values, and folds in a tuple

    Returns
    -------
    allResults : TYPE
        DESCRIPTION.

    """
    allResults=[]
    for clf, params in clfs.items():
        results = runGrid(a_clf=clf,data=data,params=params) #run the grid with the selected classifier
        bestAcc = max([r['meanAccuracy'] for r in results]) #find the best accurancy within the differnt parameters
        bestAccId = ([r['paramsId'] for r in results if r['meanAccuracy']==bestAcc])[0] #find the ID of the best accuracy
        #save everything in a dictionary
        allResults.append({'model':clf.__name__
                           ,'results':results  #results is a dictionary with the output per each parameter set
                           ,'bestMeanAccuracy':bestAcc
                           ,'bestMeanAccurancId':bestAccId
                           })
        return allResults
#%%
def runGrid(a_clf, data,params={}):
    """
    Function that acually runs each individual grid search
    
    Parameters
    ----------
    a_clf : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    params : TYPE, optional
        DESCRIPTION. The default is {}.

    Returns
    -------
    paramRes : TYPE
        DESCRIPTION.

    """
    random.seed(1701)
    M, L, n_folds = data # unpack data containter
    kf = KFold(n_splits=n_folds) # Establish the cross validation
    cv = {} # results are saved in a dictionaty
    paramGrid = list(cartesian(params)) #create the list with the cartesian product of all the passed paramters
    paramRes = [] #list that will contain the results of each parameter set.
    print("-> running model:",a_clf.__name__)
    scaler = preprocessing.StandardScaler()
    for paramId, param in enumerate(paramGrid): #run each parameter set
        print("----> param:",param,end='' )
        cv = {} # results are saved in a dictionaty
        for ids, (train_index, test_index) in enumerate(kf.split(M, L)): #run by CV
            clf = a_clf(**param) # unpack paramters into clf is they exist
            Xtrain = scaler.fit_transform(M[train_index])
            Xtest = scaler.fit_transform(M[test_index])
            clf.fit(Xtrain, L[train_index])
            pred = clf.predict(Xtest)
			#create a dictionary with the ouput of each CV
            cv[ids]= {'clf': clf  
                      ,'accuracy': accuracy_score(L[test_index], pred)
                      ,'param':param
                      ,'train_index': train_index
                      ,'test_index': test_index
                      }
        #save the output of all CV in a dictionary
        #save also the mean accuracy of all the CV
        acc=  np.mean([r['accuracy'] for r in cv.values()])
        paramRes.append({'paramsId':paramId
                         ,'meanAccuracy':  acc
                         ,'params':param
                         ,'CV':cv
                         })
        print(" = Accuracy:",acc)
    return paramRes

#%%

#Load the data
cancer = datasets.load_breast_cancer()

#M is our X data
M = np.array(cancer['data'])

#L is our Y data
L = np.array(cancer['target'])

#Set folds to 5
n_folds = 5

#Combine data into one object
data = (M, L, n_folds)

#See what we have
print(data)
#%%
#the classifieres are stored in a dictiornaty including their specific hyperparamters and selected values
clfs = {RandomForestClassifier: 
            {'n_estimators':[1,4,8]
            ,'max_depth':[10,11,12]
            ,'min_samples_split':[5,10,15]},
        svm.SVC:
            {'C':[1,5,10]
			,'kernel':['linear','rbf','poly']
            ,'tol':[1e-3,1e-1,1e-3]
			,'gamma':['auto']},
        KNeighborsClassifier:
            {'n_neighbors' :[3,5,9]
            ,'algorithm': ['ball_tree','kd_tree','brute']
            ,'p' : [1,2]}
}
    
print(clfs)
 #%%   
results = gridSearch(clfs, data) #run the codes

print('completed, output in the dictionary: results')
#the resutls are store in result dictionary.
#this dictionary contain 1 row per each model, 
#each model's row contain a dictionary with the best value,it's ID,
#and the outputs per each parameter-set saved as dictionary
#each parameter-set contains also a dictionary per each CV

#%% Organize Output List

plotX=[]
plotXmean=[]
plotY=[]
for m in results:
	for r in m['results']:
		plotX.append((list(cv['accuracy'] for cv in r['CV'].values())))
		plotXmean.append(r['meanAccuracy'])
		plotY.append(m['model'] + ':\n ' + ','.join("{!s}={!r}".format(key,val) for (key,val) in r['params'].items()))
		
sortIdx = list(np.argsort(plotXmean))

plotXmean = [plotXmean[i] for i in sortIdx]       
plotX = [plotX[i] for i in sortIdx]       
plotY = [plotY[i] for i in sortIdx]       

print("Best Classifier")
print(" ",plotY[-1])
print(" Mean Accuracy:",plotXmean[-1])

#%% 
import matplotlib.pyplot as plt

#PLOT
#set figure dimension (based on the number of eleemnts we have)
fig, ax = plt.subplots(figsize=(15,len(plotXmean)*.4))

#setting size of labels
ax.yaxis.label.set_size(20)

#axis labels
ax.set_ylabel("Parameters")
ax.set_xlabel("Accuracy")
ax.tick_params(labelsize=8)
#
fig.subplots_adjust(left=0.4,right=0.9,top=0.95,bottom=0.1)
#box plot
ax.boxplot(plotX,vert=False,labels=plotY,showmeans=True) 

#adding text for the mean point
for id,Xmean in enumerate(plotXmean):
	ax.annotate(round(Xmean,3),xy=(Xmean*1.01,id+0.87),xycoords='data',fontsize='small',color='green')

#saving to file
plt.savefig(fname = 'plot.png')#,bbox_inches='tight')

plt.show()

#%%
#flipping the order of the list to print the ranking
sortIdx = list(reversed(np.argsort(plotXmean)))
plotXmean = [plotXmean[i] for i in sortIdx]       
plotX = [plotX[i] for i in sortIdx]       
plotY = [plotY[i] for i in sortIdx]   

# PARAMETERS
printTop = 999 #<<-- Change this to limit the output list
printToFile = "N" #<<-- Use "N" to write to console, or "Y" to output to ranking_output.txt file
#
if(printToFile != "Y"):
    f = sys.stdout
else:
    filename='outputRanking.txt'
    if os.path.exists(filename):   os.remove(filename)
    f=open(filename, "a")

print("Results from other classifiers and hyperparameters, from the lowest to the highest Mean Accurancy:",file=f)
#loop to write the classifiers ranked from top to bottom
for id, r in enumerate(plotY):
    print("_______________________________",file=f)
    print("Ranking:", id+1,file=f)
    print("Classifier and Parameters",file=f)
    print("",r,file=f)
    print("    Mean Accuracy:",round(plotXmean[id],3),file=f)
    if (id+1) >= printTop: break 

if(printToFile == "Y"): f.close()

#%%
for clf in clfs.items():
    print(clf)