%% == FINE-TUNING ==

%% init dataset

close all force
trainDatasetPath = fullfile('dataset', 'train');
testDatasetPath = fullfile('dataset', 'test');
trainImgs = imageDatastore(trainDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testSet = imageDatastore(testDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% split in training and validation sets : 85% - 15%
quotaForEachLabel = 0.85;
[trainSet, validationSet] = splitEachLabel(trainImgs, quotaForEachLabel, 'randomize');

%% net config

net = alexnet;

inputSize = net.Layers(1).InputSize;

layersTransfer = net.Layers(1:end-3);

layers = [
    
    layersTransfer
    
    fullyConnectedLayer(15, 'WeightLearnRateFactor', 50, 'BiasLearnRateFactor', 50)
    softmaxLayer
    classificationLayer
    
    ];


%% img pre-processing

% from 1 channel to 3 channel
trainSet.ReadFcn = @(x)repmat(imread(x), 1, 1, 3);
validationSet.ReadFcn = @(x)repmat(imread(x), 1, 1, 3);
testSet.ReadFcn = @(x)repmat(imread(x), 1, 1, 3);

augmenter = imageDataAugmenter('RandXReflection', true);

% resizing
augTrainSet = augmentedImageDatastore(inputSize(1:2), trainSet);
augValidationSet = augmentedImageDatastore(inputSize(1:2), validationSet);
augTestSet = augmentedImageDatastore(inputSize(1:2), testSet);


%% net options

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', augValidationSet, ...
    'MaxEpochs', 15, ...
    'ValidationPatience', 5, ...
    'ExecutionEnvironment', 'parallel', ...
    'Plots', 'training-progress')

%% training

netTransfer = trainNetwork(augTrainSet, layers, options);

%% test

YPredicted = classify(netTransfer, augTestSet);
YReal = testSet.Labels;

% overall accuracy
accuracy = sum(YPredicted == YReal) / numel(YReal)

% confusion matrix
figure
plotconfusion(YReal, YPredicted)
