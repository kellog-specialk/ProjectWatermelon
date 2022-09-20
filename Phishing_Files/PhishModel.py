"""

    Necessary Packages:
    * Tensorflow 
    * Pandas 
    * Scikit-learn

"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import save_model
from datetime import datetime 

tf.random.set_seed(18)

# * converts text input to binary 1 and 0
# * used to convert y column of data from 
#   "phishing" and "legitimate"
#   to 1 for phishing and 0 for legitimate
def textToBinary(txt):
    if txt=="phishing":
        return 1
    else:
        return 0

if __name__ == '__main__':
    
    # load the dataset
    global root_directory
    root_directory = '/Users/blumezl1/Documents/ProjectWatermelon/'


    pdDataset = pd.read_csv(root_directory + "Phish_Full_Dataset.csv")
    pdDataset = pdDataset.drop(columns=['url'])


    # change output data to neural network friendly 1s and 0s
    pdDataset['status'] = pdDataset['status'].apply(lambda x: textToBinary(x))


    # combine columns (use as needed):
    # pdDataset['newColumn'] = PdDataset['col1'] + PdDataset['col2']


    # split into input (X) and output (y) variables    
    # with numpy array indexing
    dataset = pdDataset.to_numpy()
    X = dataset[:,0:87]
    y = dataset[:,87]


    
    # preprocessing using sklearn
    scalar = StandardScaler()
    X = scalar.fit_transform(X)

    
    
    #split into test and training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # define the keras model
    model = Sequential()
    # first hidden layer based off input
    model.add(Dense(12, input_shape=(87,), activation='relu'))
    # hidden
    for i in range(2):
        model.add(Dense(12, activation='relu'))
    #output
    model.add(Dense(1, activation='sigmoid'))


    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    # fit the keras model on the dataset
    model.fit(X_train, y_train, epochs=10, batch_size=18, verbose=0)


    # evaluate the keras model
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("*************************")
    print('Accuracy: %.2f' % (accuracy*100))
    print("*************************")
    # get time that model was evaluted at
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


    # save model under a certain filename
    save_model(model,root_directory + "PhishPipelinev2.h5")

    