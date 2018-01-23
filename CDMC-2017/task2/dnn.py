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

from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dense, Dropout
from keras import callbacks
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.utils.np_utils import to_categorical

traindata = pd.read_csv('train.csv', header=None)
testdata = pd.read_csv('test.csv', header=None)

X = traindata.iloc[:,1:14]
Y = traindata.iloc[:,0]
T = testdata.iloc[:,0:14]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)


train_matrix = np.array(trainX)
trainlabel = np.array(Y)
trainlabel = to_categorical(trainlabel)
test_matrix = np.array(testT)




model = Sequential()
model.add(Dense(100, input_dim=12, activation='relu'))
model.add(Dropout(.01))
model.add(Dense(100, activation='relu'))
model.add(Dropout(.01))
model.add(Dense(100, activation='relu'))
model.add(Dropout(.01))
model.add(Dense(100, activation='relu'))
model.add(Dropout(.01))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="logs/dnn4layer/checkpoint-{epoch:02d}.hdf5", verbose=1,save_best_only=True, monitor='loss')
csv_logger = CSVLogger('logs/dnn4layer/dnntrainanalysis.csv',separator=',', append=False)
model.fit(train_matrix, trainlabel, nb_epoch=1000, batch_size=128, callbacks=[checkpointer,csv_logger])



