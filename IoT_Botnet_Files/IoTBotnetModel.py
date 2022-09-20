# packages used
import pandas as pd
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import save_model, load_model
import tensorflow as tf

tf.random.set_seed(20)

if __name__ == '__main__':

        # load the dataset from projectwatermelon directory
        global root_directory
        root_directory = '/Users/chowjw1/Documents/ProjectWatermelon/'
        
        # read in the dataset 
        df = pd.read_csv('IoTBotnetData.csv', delimiter=',')
        
        # splicing the x and y output in dataset
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # split dataset to testing and training data
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=0)

        
        model = None
        use_saved_model = True
        
        if use_saved_model:
                model = load_model(root_directory + "IoTSplit.h5")
                

        # calling saved model
        else:

                # define the keras model
                model = Sequential()
                # first hidden layer based off input
                model.add(Dense(12, input_shape=(115,), activation='relu'))
                # hidden layer
                model.add(Dense(8, activation='relu'))
                # output layer
                model.add(Dense(1, activation='sigmoid'))
                # compile the keras model
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


                ## fit the keras model on the dataset
                model.fit(x_train, y_train, epochs=10, batch_size=10)


        ## evaluate the keras model
        ## print accuracy rate

        # _, accuracy = model.evaluate(x_test, y_test)
        # results = model.evaluate(x_test, y_test)
        # print(results) 

        ## prints machine predictions
        results = model.predict(
                x,
                batch_size=None,
                verbose="auto",
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False,
        )

        print(results)
        # results = pd.to_csv(root_directory, + "Results")
        
        # print("\n\n\n*********************************")
        # print('Accuracy: %.2f' % (accuracy*100))
        # print("*********************************")
        

        ## print x_test to csv file
        # prediction = pd.DataFrame(y_test, model.evaluate(y_test))
        # prediction.to_csv("yTestsEval.csv")

        ## save model under a filename
        # save_model(model, root_directory + "IoTSplit.h5")
        # print(model, x_test, y_test)


