# ProjectWatermelon

A summer intern project created by Z. Blumenfeld and J. Chow under the supervision of K. Zaback

##  Description

ProjectWatermelon was designed to establish a neural network based on topics related to cybersecurity.

It is comprised of two main cyber-applicable subsections - phishing website links and IoT Botnet attacks. 

Both subsections are located in their respective directories and contain the following: the dataset(s) used to train the neural networks, the python scripting also used in training, and the pipelines saved as .h5 files.


## Applications used:
* Package manager - Anaconda 4.13.0
* Main machine learning library - [Keras](https://keras.io)/[Tensorflow](https://www.tensorflow.org)
* Some specific packages (check code for more) - pandas, scikit-learn, numpy, and packages from Tensorflow


## Datasets
The phishing dataset used can be located [here](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset), and the botnet dataset used can be located [here](https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT)

# Important to Note

About the phishing data: 

The different attributes for phishing data are obtained by running a series of algorithms on any specific URL. Those algorithms are written in python can be found in the scripts folder [here](https://data.mendeley.com/datasets/c2gw7fy2j4/3). The data is also preprocessed using sklearn in PhishingModel.py prior to compiling/training, also within the same file. 


About the botnet data:

Only data from 1 out of 9 available IoT devices was used from the [Botnet dataset](https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT)

## Roadmap
The pipelines created in this project are intended to be used in an AI explanability algorithm that would add additional human readibility for the project.


## Authors and acknowledgment
Authored by: Zachary Blumenfeld and Joycelyn Chow

A special thanks to Katie Zaback and Morgan Mango for mentoring and helping out greatly with the development of this project. 

A big thank you to APL's ASPIRE program and staffing for facilitating the opportunity to work on this project.


## Project status
Done

Project is still maintained by Zachary Blumenfeld and Joycelyn Chow, and consequently may undergo minor changes. The goal and project itself, however, are completed.

## Additional information
A write up of this project is available in the ProjectDescription.doc file.

For any further questions or concerns, please reach out: 

* Z. Blumenfeld - zachb0077 '@' gmail.com or  Zach.Blumenfeld '@' jhuapl.edu
* J. Chow - Joycelyn.Chow '@' jhuapl.edu