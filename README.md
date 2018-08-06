There are 400K Amazon reviews in csv file: https://drive.google.com/open?id=1bDwcBdCiEZ2pfLOc6ANi85TLAspyhYUR. I just picked every 5th review as test data and the rest of them as traning data. First I use **scikit-learn** to get the feature matrix of our data, then use 3 methods including Decision Tree, Neural Network and Naive Bayes to do prediction. I also used **pyplot** to plot some curves which shown the performance of these 3 methods. 
<img src="https://user-images.githubusercontent.com/11751622/43691416-030fdc58-98d1-11e8-8416-593bddfa6625.png" width="2000" height="600">
<img src="https://user-images.githubusercontent.com/11751622/43691378-3ed3a7fc-98d0-11e8-83e8-8a5b3550e21a.png" width="430" height="430">
<img src="https://user-images.githubusercontent.com/11751622/43691381-4069fd14-98d0-11e8-8cb0-dfbbbbdabcbb.png" width="430" height="430">

According to these curves, we can conclude that, if we have enough data (int our example, it should be more than 200k) to train our model, we should use Neural Network. So I produced my prediction model: model.pkl, using Neural Network.
