# Machine-Learning-Sentiment-Analysis
Machine Learning Sentiment Analysis Training, Tweets.

 

User notice
--

- (1st) Run sentiment analysis script first.
- (2nd) Once Preprocessing has finished run Model scripts.

Also remember to change file paths with where the dataset files reside.


---------

Notes: 

- need to add try and catch blocks for error handling if any errors occur.
- Adding attention layer for new results and modifiying attention layer.
- Using Gru layers and standard LSTM layers.
- changing exponential learning decay rates for further analysis and   Learning Rate schedules, will further accomadate for over fitting and under fitting and majority biases.

- improved model performance via optimisation of shuffling and using 20% of the dataset, also additional hyper parameter tuning.

- majority and minority class sizes are now balanced and nan values dealt with through mean relation to non nan features.
- prevented overfitting and underfitting taking into account loss via learning rates due to undershooting optimal solutions and experiencing lower convergence.

- removed dropout layers to prevent underfitting.

Most Recent Update: 
---
- further reduced underfitting and now using sgdm with batch gradient descent to now have the model working correctly with less fluctuations, escaping low non optimal local minima's and saddle points as shown by accuracy after the inital 18 epochs, increasing from 20% at the global minima point to 60% by the 42nd epoch with stable accuracy momentum after the inital global minima point reached at the 18th epoch.

- training with 3 bilstm layers for first test, 2nd test with 3 gru layers, 3rd test with 3 standard lstm layers.

Future Implementation:
----
- need to create custom dot product attention layer.
- need to create dynamic training loop for further optimised training compared to Adam or sgdm.
- need to use pretrained model training data for further training. 

-----
RNN Logic:

- 
-
-


Model Layers: 
---

- LSTM : layers[]

- GRU : layers[]


- GRU-LSTM : layers[]


- BiLSTM : layers[]
-----
Training results:  



Result Conclusion : BiLSTM is better for SA tasks.

-----
Sentiment Training Logic and functions:

-
-
-

-----

Dataset url: https://www.kaggle.com/datasets/mukulkirti/positive-and-negative-word-listrar


Positive & Negative words url: 
https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset?resource=download&select=test.csv 


-----
Requires:
- minimum 23.9gb ram (24 gb of ram preferable) could try reducing hidden units and Initial learning rate so less learnable's being used so less ram is required however will affect result accuracy (wouldn't recommend decreasing learning rate below 0.1).

- MATLAB add-ons specified.
-----
Matlab Addon:

- Deep Learning ToolBox
- Text Analytics ToolBox
- Statistics and Machine Learning Toolbox


Issues
--
- BGD requires a lot of memory for visual analysis
- GPU memory was limited when collecting other results than model validation and mean accuracy.
- BGD allowed for increased learning and calibration of neurons in each layer.
