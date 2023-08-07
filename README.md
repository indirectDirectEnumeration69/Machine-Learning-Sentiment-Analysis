# Machine-Learning-Sentiment-Analysis
Machine Learning Sentiment Analysis Training, Tweets.

 
Notes: 

- need to add try and catch blocks for error handling if any errors occur.
- Adding attention layer for new results and modifiying attention layer.
- Using Gru layers and standard LSTM layers.
- changing exponential learning decay rates for further analysis and   Learning Rate schedules, will further accomadate for over fitting and under fitting and majority biases.

- improved model performance via optimisation of shuffling and using 20% of the dataset, also additional hyper parameter tuning.

- majority and minority class sizes are now balanced and nan values dealt with through mean relation to non nan features.
- prevented overfitting and underfitting taking into loss via learning rates due to undershooting optimal solutions and experiencing lower convergence.



Most Recent Update: 
---
- further reduced underfitting and now using sgdm with batch gradient descent to now have the model working correctly and accuracy after the inital 18 epochs increasing from 30% to 60% after the 42nd epoch with stable accuracy momentum.

training with 3 bilstm layers for first test, 2nd test with 3 gru layers, 3rd test with 3 standard lstm layers.
-----
RNN Logic:

-
-
-
-----
Training results:

-----
Sentiment Training Logic and functions:

-
-
-

-----

Dataset url:


Positive & Negative words url:


-----
Requires:
- minimum 23.9gb ram (24 gb of ram preferable) could try reducing hidden units and Initial learning rate so less learnables being used so less ram required however will affect result accuracy (wouldnt recommend decreasing learning rate below 0.1).

-
-
-----
Matlabs Addon:

-
-
-
