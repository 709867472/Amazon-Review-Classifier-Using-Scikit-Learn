# Amazon-Review-Classifier-Using-Scikit-Learn
We have 400K Amazon reviews in csv file: https://drive.google.com/open?id=1bDwcBdCiEZ2pfLOc6ANi85TLAspyhYUR. I just picked every 5th review as test data and the rest of them as traning data. First I use scikit-learn to get the feature matrix of our data, then use 3 methods including Decision Tree, Neural Network and Naive Bayes to predict a new review. I also uesed pyplot to plot some curves which shown the performance of these 3 methods. 
<img src="https://user-images.githubusercontent.com/11751622/43679140-eb49ebe2-97d4-11e8-8f93-1a37642ded07.png" width="400" height="400">
<img src="https://user-images.githubusercontent.com/11751622/43679141-ee3d114e-97d4-11e8-80af-239a6967592e.png" width="400" height="400">

According to these curve, we can conclude that, if we have enough data to training data to train our model, we should use Neural Network. So I just used Neural Network to produce my prediction model: model.pkl.

