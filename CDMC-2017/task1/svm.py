from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import Normalizer
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)
from sklearn.decomposition import TruncatedSVD



traindata = pd.read_csv('train.csv', header=None)
testdata = pd.read_csv('test.csv', header=None)

X = traindata.iloc[:,1:4000]

T = testdata.iloc[:,0:4000]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)


traindata = np.array(trainX)



testdata = np.array(testT)

TRAINDATA = traindata

def Splitdata(TRAINDATA):
    TRAINDATA1 = []
    TRAINDATA2 =[]
    Trainlabel1 = []
    Trainlabel2 = []
    [r1,c1] = TRAINDATA.shape
    
    for i in range(0,r1):
        if (TRAINDATA[i,0] == 1):
            TRAINDATA1.append(TRAINDATA[i,1:c1])
            Trainlabel1.append(TRAINDATA[i,0])
        if(TRAINDATA[i,0] == 0):
            TRAINDATA2.append(TRAINDATA[i,1:c1])
            Trainlabel2.append(TRAINDATA[i,0])
   
    return TRAINDATA1, TRAINDATA2, Trainlabel1, Trainlabel2

TRAINDATA1, TRAINDATA2, Trainlabel1, Trainlabel2 = Splitdata(TRAINDATA)
def CROSSVALIDATION(TRAINDATA1,TRAINDATA2,Trainlabel1,Trainlabel2,Maxnocomp,step,randomstatemax):
    
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import TruncatedSVD, NMF
    from sklearn.model_selection import cross_val_score
    from sklearn import svm
    from sklearn.svm import SVC
    
    
    Value = 0
    AvgAcc = 0
        
    for j in range(10,Maxnocomp,step):    
        svd = TruncatedSVD(n_components=j, n_iter=7, random_state=42)
        SVD_Matrix1 = svd.fit_transform(TRAINDATA1)
        SVD_Matrix2 = svd.fit_transform(TRAINDATA2)
        
        for i in range(0,randomstatemax):
            
            X_train1, X_test1, y_train1, y_test1 = train_test_split(SVD_Matrix1,Trainlabel1, test_size=0.05, random_state=i)
            X_train2, X_test2, y_train2, y_test2 = train_test_split(SVD_Matrix2,Trainlabel2, test_size=0.05, random_state=i)
          
            X_train = np.append(X_train1,X_train2,axis = 0)
            y_train = np.append(y_train1,y_train2,axis = 0)
            
            X_test = np.append(X_test1,X_test2,axis = 0)
            y_test = np.append(y_test1,y_test2,axis = 0)
            clf = SVC(kernel='linear')
            clf.fit(X_train, y_train)
            Value = clf.score(X_test,y_test)*100
            AvgAcc = AvgAcc + Value
            print '-----No of Components:',j,'-----Random_State:',i
            print 'Accuracy in %', Value    
        print'#############'    
        print '--AVERAGE ACCURACY',AvgAcc/10
        AvgAcc = 0
	#X_train = np.zeros(X_train)
	#y_test = np.zeros(y_test)
	#X_test = np.zeros(X_test)
	#y_train = np.zeros(y_train)
                
    
CROSSVALIDATION(TRAINDATA1,TRAINDATA2,Trainlabel1,Trainlabel2,Maxnocomp=40,step=10,randomstatemax=10)



