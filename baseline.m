%% variables

clear
full = 0;
imgSize = [64 64];


%% load dataset

close all force
trainDatasetPath = fullfile('dataset', 'train');
testDatasetPath = fullfile('dataset', 'test');
trainImgs = imageDatastore(trainDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testSet = imageDatastore(testDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% split in training and validation sets : 85% - 15%
quotaForEachLabel = 0.85;
[trainSet, validationSet] = splitEachLabel(trainImgs, quotaForEachLabel, 'randomize');

%% image pre-processing

% automatic resizing
trainSet.ReadFcn = @(x)imresize(imread(x), imgSize);
validationSet.ReadFcn = @(x)imresize(imread(x), imgSize);
testSet.ReadFcn = @(x)imresize(imread(x), imgSize);


%% analyze dataset

if full == 1
    
    %labelCount = countEachLabel(trainSet)
    %unique(trainSet.Labels)
    
    % show some instances of taining set
    figure;
    perm = randperm(length(trainSet.Labels),20) ;
    for ii = 1:20
        subplot(4, 5, ii);
        imshow(trainSet.Files{perm(ii)});
        title(trainSet.Labels(perm(ii)));
    end
    sgtitle('some instances of the training set');
    
end

%% init network

layers = [
    
    imageInputLayer([imgSize(1) imgSize(2) 1], 'Name', 'input')
    convolution2dLayer(3 ,8 , 'Padding', 'same', 'WeightsInitializer', 'narrow-normal', 'Bias', zeros(1, 1, 8), 'Name', 'conv_1')
    reluLayer('Name', 'relu_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')
    
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv_2')
    reluLayer('Name','relu_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_3')
    reluLayer('Name', 'relu_3')
    
    fullyConnectedLayer(15, 'Name', 'fc_1')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.0005, ...
    'ValidationData', validationSet, ...
    'ValidationPatience', 5, ...
    'MiniBatchSize', 32, ...
    'ExecutionEnvironment', 'parallel',...
    'Plots', 'training-progress')

%% analyze structure

if full == 1
    
    lgraph = layerGraph(layers);
    analyzeNetwork(lgraph)
    
end

%% TRAINING

net = trainNetwork(trainSet, layers, options);

%% TEST

YPredicted = classify(net, testSet);
YReal = testSet.Labels;

% overall accuracy
accuracy = sum(YPredicted == YReal) / numel(YReal)

% confusion matrix
figure
plotconfusion(YReal, YPredicted)

