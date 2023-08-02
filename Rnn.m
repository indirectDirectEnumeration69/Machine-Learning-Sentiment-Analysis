% Load the preprocessed data
load('preprocessedData.mat');

% Convert labels to categorical
YTrain = categorical(YTrain);
YValidation = categorical(YValidation);
YTest = categorical(YTest);

% Determine the minority class (now that YTrain is categorical)
minorityClassCounts = countcats(YTrain);
[~, minorityClassIdx] = min(minorityClassCounts);
minorityClassLabel = categories(YTrain);
minorityClass = minorityClassLabel(minorityClassIdx);

% Find indices of the minority class
minorityClassIndices = find(YTrain == minorityClass);

% Define the number of oversamples
numberOfOversamples = 200;

% Randomly select indices from the minority class to oversample
randomIndices = randi(numel(minorityClassIndices), 1, numberOfOversamples);

% Get the oversampled data and labels
oversampledMinorityData = XTrain(minorityClassIndices(randomIndices));
oversampledMinorityLabels = YTrain(minorityClassIndices(randomIndices));

% Concatenate the oversampled data to the original data
XTrain = [XTrain, oversampledMinorityData];
YTrain = [YTrain; oversampledMinorityLabels];

% Define model parameters
inputSize = size(XTrain{1}, 1);
numHiddenUnits = 200; 
numClasses = numel(categories(YTrain));
numFeatures = size(XTrain{1}, 1);
numSequences = size(XTrain{1}, 2);
maxEpochs = 100;
miniBatchSize = 20; 
initialLearnRate = 0.05; 
decay = 0.95;

% Convert labels to categorical
YTrain = categorical(YTrain);
YValidation = categorical(YValidation);
YTest = categorical(YTest);

attentionLayer = AttentionLayer("Attention Layer", numHiddenUnits);

disp('Formatting checks while loading...')
disp('Number of Classes:');
disp(numClasses);
disp('Size of the first input sequence:');
disp(size(XTrain{1}));
disp('Size of the YTrain:');
disp(size(YTrain));

% Define the network structure
layers = [
    sequenceInputLayer(inputSize, 'Name','input')
    bilstmLayer(numHiddenUnits, 'OutputMode', 'sequence', 'Name', 'bilstm1')
    batchNormalizationLayer('Name','bn1')
    dropoutLayer(0.4, 'Name','drop1')
    bilstmLayer(numHiddenUnits, 'OutputMode','last', 'Name','bilstm2')
    batchNormalizationLayer('Name','bn2')
    dropoutLayer(0.4,'Name','drop2')
    fullyConnectedLayer(numClasses, 'Name','fc', 'WeightL2Factor', 0.001)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];
      
analyzeNetwork(layers)

epochsDrop = 1:maxEpochs;
learnRateDropFactor = decay;

% Training options
options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Verbose',0, ...
    'Plots','training-progress', ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',30, ...
    'Shuffle','every-epoch', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor', decay, ...
    'LearnRateDropPeriod',20, ...
    'InitialLearnRate',initialLearnRate, ...
    'ValidationPatience', inf);


% Train the model
net = trainNetwork(XTrain, YTrain, layers, options);

% Classify the training set and calculate accuracy
YPred_train = classify(net, XTrain);
accuracy_train = mean(YPred_train == YTrain);
fprintf('Training accuracy: %f\n', accuracy_train);

% Classify the validation set and calculate accuracy
YPred_validation = classify(net, XValidation);
accuracy_validation = mean(YPred_validation == YValidation);
fprintf('Validation accuracy: %f\n', accuracy_validation);

% Classify the testing set and calculate accuracy
YPred_test = classify(net, XTest);
accuracy_test = mean(YPred_test == YTest);
fprintf('Testing accuracy: %f\n', accuracy_test);

% Create a confusion matrix for the test set
figure
cm = confusionmat(YTest, YPred_test);
heatmap(cm);
title('Confusion Matrix for Test Data');

% Print Precision, Recall, F1 Score for the test set
[precision, recall, f1Score] = calculateMetrics(YTest, YPred_test);
fprintf('Test Precision: %f, Recall: %f, F1 Score: %f\n', precision, recall, f1Score);

% Define the function to calculate Precision, Recall, F1 Score
function [precision, recall, f1Score] = calculateMetrics(YTrue, YPred)
    % Convert categorical variables to numerical for calculations
    YTrue = grp2idx(YTrue);
    YPred = grp2idx(YPred);
    
    % Create a confusion matrix
    cm = confusionmat(YTrue, YPred);
    
    % Calculate precision, recall, f1 score
    tp = diag(cm); % True positive
    fp = sum(cm, 1)' - tp; % False positive
    fn = sum(cm, 2) - tp; % False negative
    
    precision = mean(tp ./ (tp + fp + 1e-10));
    recall = mean(tp ./ (tp + fn + 1e-10));
    f1Score = 2 * (precision * recall) / (precision + recall);
end
