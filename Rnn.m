load('preprocessedData.mat');

% Display NaN locations.
displayNaNLocations(XTrain, 'XTrain');
displayNaNLocations(XValidation, 'XValidation');
displayNaNLocations(XTest, 'XTest');

% Now replacing the NaN values with the mean
XTrain = replaceNaNWithMean(XTrain);
XValidation = replaceNaNWithMean(XValidation);
XTest = replaceNaNWithMean(XTest);

% Check NaN values in the data
if any(cellfun(@(x) any(isnan(x(:))), XTrain)) || any(isnan(YTrain(:))) || ...
   any(cellfun(@(x) any(isnan(x(:))), XValidation)) || any(isnan(YValidation(:))) || ...
   any(cellfun(@(x) any(isnan(x(:))), XTest)) || any(isnan(YTest(:)))
    error('NaN values found in the data, clean the data before training.');
end

% labels to categorical
YTrain = categorical(YTrain);
YValidation = categorical(YValidation);
YTest = categorical(YTest);

% majority class size
majorityClassSize = max(countcats(YTrain));

% balanced training data
balancedXTrain = cell(majorityClassSize * numel(categories(YTrain)), 1);
balancedYTrain = categorical();
counter = 1;

for classLabel = categories(YTrain)'
    classIndices = find(YTrain == classLabel{1});
    resampledIndices = datasample(classIndices, majorityClassSize, 'Replace', true); % Oversample with replacement
    balancedXTrain(counter:counter + majorityClassSize - 1) = XTrain(resampledIndices);
    balancedYTrain(counter:counter + majorityClassSize - 1, 1) = YTrain(resampledIndices);
    counter = counter + majorityClassSize;
end

% Shuffle the balanced training data
randOrder = randperm(majorityClassSize * numel(categories(YTrain)));
XTrain = balancedXTrain(randOrder);
YTrain = balancedYTrain(randOrder);

% model parameters
inputSize = size(XTrain{1}, 1);
numHiddenUnits = 200; 
numClasses = numel(categories(YTrain));
maxEpochs = 100;
fullBatchSize = numel(YTrain);
miniBatchSize = fullBatchSize; 
initialLearnRate = 0.1; 
decay = 0.85;

disp('Formatting checks while loading...');
disp('Number of Classes:');
disp(numClasses);
disp('Size of the first input sequence:');
disp(size(XTrain{1}));
disp('Size of the YTrain:');
disp(size(YTrain));

% Reduced dropouts for underfitting issue.
layers = [
    sequenceInputLayer(inputSize, 'Name','input')
    bilstmLayer(numHiddenUnits, 'OutputMode', 'sequence', 'Name', 'bilstm1')
    batchNormalizationLayer('Name','bn1')
    bilstmLayer(numHiddenUnits, 'OutputMode', 'sequence', 'Name', 'bilstm2') % Add a new bilstm layer
    batchNormalizationLayer('Name','bn2')
    bilstmLayer(numHiddenUnits, 'OutputMode','last', 'Name','bilstm3') % Make the last bilstm layer output only the last sequence
    batchNormalizationLayer('Name','bn3')
    fullyConnectedLayer(numClasses, 'Name','fc', 'WeightL2Factor', 0.001)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];

analyzeNetwork(layers)

epochsDrop = 1:maxEpochs;
learnRateDropFactor = decay;

% options
options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Verbose',0, ...
    'Plots','training-progress', ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',2, ...
    'Shuffle','every-epoch', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor', decay, ...
    'LearnRateDropPeriod',7, ...
    'InitialLearnRate',initialLearnRate, ...
    'ValidationPatience', inf);

net = trainNetwork(XTrain, YTrain, layers, options);


%requires alot of gpu memory.

% training set and calculate accuracy
YPred_train = classify(net, XTrain);
accuracy_train = mean(YPred_train == YTrain);
fprintf('Training accuracy: %f\n', accuracy_train);

% validation set and calculate accuracy
YPred_validation = classify(net, XValidation);
accuracy_validation = mean(YPred_validation == YValidation);
fprintf('Validation accuracy: %f\n', accuracy_validation);

% testing set and calculate accuracy
YPred_test = classify(net, XTest);
accuracy_test = mean(YPred_test == YTest);
fprintf('Testing accuracy: %f\n', accuracy_test);

% Create a confusion matrix for the test set
figure
cm = confusionmat(YTest, YPred_test);
classNames = categories(YTest); 
heatmap(cm, classNames, classNames, 1, 'Colormap', 'cool', ...
        'Colorbar', true, 'ShowAllTicks', true, ...
        'XLabel', 'Predicted', 'YLabel', 'Actual');
title('Confusion Matrix for Test Data');

% Print Precision, Recall, F1 Score for the test set
[precision, recall, f1Score] = calculateMetrics(YTest, YPred_test);
fprintf('Test Precision: %f, Recall: %f, F1 Score: %f\n', precision, recall, f1Score);

% function to calculate Precision, Recall, F1 Score
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

% Function to replace NaN values with the mean of non-NaN values for each feature
function cellData = replaceNaNWithMean(cellData)
    % Determine the size of the feature
    featureSize = size(cellData{1}, 1);
    
    % Calculate the mean for each feature, ignoring NaN values
    featureMeans = zeros(featureSize, 1);
    for i = 1:featureSize
        featureValues = cellfun(@(x) mean(x(i, 'omitnan')), cellData);
        featureMeans(i) = mean(featureValues, 'omitnan');
    end

    % replacing NaN values with the corresponding feature mean
    for i = 1:numel(cellData)
        data = cellData{i};
        for j = 1:featureSize
            data(j, isnan(data(j, :))) = featureMeans(j);
        end
        cellData{i} = data;
    end
end

function displayNaNLocations(cellData, dataName)
    fprintf('Checking for NaN values in %s...\n', dataName);
    for i = 1:numel(cellData)
        data = cellData{i};
        [rows, cols] = find(isnan(data));
        if ~isempty(rows)
            fprintf('NaN found in sequence %d at the following positions (row, col):\n', i);
            disp([rows, cols]);
        end
    end
    fprintf('Finished checking for NaN values in %s.\n', dataName);
end



