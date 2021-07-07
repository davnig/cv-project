%% variables

clear
full = 0;
imgSize = [64 64];

improve = 1; % baseline + data augmentation
improve = 2; %          + batch normalization
improve = 3; %          + increased number and size of filters
improve = 4; %          + more conv layers


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


%% 1. + DATA AUGMENTATION (left-to-right reflection)

aug = imageDataAugmenter('RandXReflection', true);
augTrainSet = augmentedImageDatastore(imgSize, trainSet, 'DataAugmentation', aug);

if improve == 1
    
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', 0.0005, ...
        'ValidationData', validationSet, ...
        'ValidationPatience', 5, ...
        'ExecutionEnvironment', 'parallel', ...
        'Plots', 'training-progress')
    
    layers = [
        imageInputLayer([imgSize(1) imgSize(2) 1], 'Name', 'input')
        
        convolution2dLayer(3, 8, 'Padding', 'same', 'WeightsInitializer', 'narrow-normal', 'Bias', zeros(1, 1, 8), 'Name', 'conv_1')
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
    
end

%% 2. + BATCH NORMALIZATION

if improve == 2
    
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', 0.001, ...
        'ValidationData', validationSet, ...
        'ValidationPatience', 5, ...
        'ExecutionEnvironment', 'parallel', ...
        'Plots', 'training-progress')
    
    layers = [
        imageInputLayer([imgSize(1) imgSize(2) 1], 'Name', 'input')
        
        convolution2dLayer(3, 8, 'Padding', 'same', 'WeightsInitializer', 'narrow-normal', 'Bias', zeros(1, 1, 8), 'Name', 'conv_1')
        batchNormalizationLayer('Name', 'BN_1')
        reluLayer('Name', 'relu_1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')
        
        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv_2')
        batchNormalizationLayer('Name', 'BN_2')
        reluLayer('Name','relu_2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')
        
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_3')
        batchNormalizationLayer('Name', 'BN_3')
        reluLayer('Name', 'relu_3')
        
        fullyConnectedLayer(15, 'Name', 'fc_1')
        softmaxLayer('Name', 'softmax')
        
        classificationLayer('Name', 'output')
        ];
    
end
    
%% 3. + INCREASED NUMBER AND SIZE OF FILTERS

if improve == 3
    
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', 0.001, ...
        'ValidationData', validationSet, ...
        'ValidationPatience', 5, ...
        'ExecutionEnvironment', 'parallel', ...
        'Plots', 'training-progress')
    
    layers = [
        imageInputLayer([imgSize(1) imgSize(2) 1], 'Name', 'input')
        
        convolution2dLayer(3, 16, 'Padding', 'same', 'WeightsInitializer', 'narrow-normal', 'Bias', zeros(1, 1, 16), 'Name', 'conv_1')
        batchNormalizationLayer('Name', 'BN_1')
        reluLayer('Name', 'relu_1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')
        
        convolution2dLayer(5, 32, 'Padding', 'same', 'Name', 'conv_2')
        batchNormalizationLayer('Name', 'BN_2')
        reluLayer('Name','relu_2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')
        
        convolution2dLayer(7, 64, 'Padding', 'same', 'Name', 'conv_3')
        batchNormalizationLayer('Name', 'BN_3')
        reluLayer('Name', 'relu_3')
        
        fullyConnectedLayer(15, 'Name', 'fc_1')
        softmaxLayer('Name', 'softmax')
        
        classificationLayer('Name', 'output')
        ];
    
end

%% 4. + MORE CONV LAYERS & DROPOUT

if improve == 4
    
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', 0.005, ...
        'ValidationData', validationSet, ...
        'ValidationPatience', 5, ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 128, ...
        'ExecutionEnvironment', 'parallel', ...
        'Plots', 'training-progress')
    
    layers = [
        imageInputLayer([imgSize(1) imgSize(2) 1], 'Name', 'input')
        
        convolution2dLayer(3, 16, 'Padding', 'same', 'WeightsInitializer', 'narrow-normal', 'Bias', zeros(1, 1, 16), 'Name', 'conv_1')
        batchNormalizationLayer('Name', 'BN_1')
        reluLayer('Name', 'relu_1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')
        
        convolution2dLayer(5, 32, 'Padding', 'same', 'Name', 'conv_2')
        batchNormalizationLayer('Name', 'BN_2')
        reluLayer('Name','relu_2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')
        
        convolution2dLayer(7, 64, 'Padding', 'same', 'Name', 'conv_3')
        batchNormalizationLayer('Name', 'BN_3')
        reluLayer('Name', 'relu_3')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_3')
        
        convolution2dLayer(9, 64, 'Padding', 'same', 'Name', 'conv_4')
        batchNormalizationLayer('Name', 'BN_4')
        reluLayer('Name', 'relu_4')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_4')
        
        convolution2dLayer(11, 64, 'Padding', 'same', 'Name', 'conv_5')
        batchNormalizationLayer('Name', 'BN_5')
        reluLayer('Name', 'relu_5')
        
        dropoutLayer()
        
        fullyConnectedLayer(15, 'Name', 'fc_1')
        softmaxLayer('Name', 'softmax')
        
        classificationLayer('Name', 'output')
        ];
    
end

%% TRAINING & TEST

net = trainNetwork(augTrainSet, layers, options);

YPredicted = classify(net, testSet);
YReal = testSet.Labels;

% overall accuracy
accuracy = sum(YPredicted == YReal) / numel(YReal)

% confusion matrix
figure
plotconfusion(YReal, YPredicted)

