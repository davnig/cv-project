%% == FEATURE EXTRACTION & SVM ==

%% Init dataset

clear
close all force
trainDatasetPath = fullfile('dataset', 'train');
testDatasetPath = fullfile('dataset', 'test');
trainImgs = imageDatastore(trainDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testSet = imageDatastore(testDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% split in training and validation sets : 85% - 15%
quotaForEachLabel = 0.85;
[trainSet, validationSet] = splitEachLabel(trainImgs, quotaForEachLabel, 'randomize');

%% Pre-process images

net = alexnet;
inputSize = net.Layers(1).InputSize;

% from 1 channel to 3 channel
trainSet.ReadFcn = @(x)repmat(imread(x), 1, 1, 3);
testSet.ReadFcn = @(x)repmat(imread(x), 1, 1, 3);

augTrainSet = augmentedImageDatastore(inputSize(1:2), trainSet);
augTestSet = augmentedImageDatastore(inputSize(1:2), testSet);

%% Extract image features

layer = 'fc7';
XTrain = activations(net,augTrainSet, layer, 'OutputAs', 'rows'); % 4096 features
XTest = activations(net, augTestSet, layer, 'OutputAs', 'rows'); % 4096 features

YTrain = trainSet.Labels;
YTest = testSet.Labels;

fprintf('Loaded dataset\n');

%% Fit Pairwise SVM

fprintf('Fitting pairwise SVM on training set...\n');

classes = unique(trainImgs.Labels);
combinations = nchoosek(classes, 2);
n_classifiers = size(combinations, 1);
SVMModels  = cell(n_classifiers, 1);

for i = 1 : n_classifiers
    
    class_i = combinations(i, 1);
    class_j = combinations(i, 2);
    
    SVMModels{i} = fitcsvm(XTrain, YTrain, 'ClassNames', [class_i, class_j]);
    
end

%% Predict

fprintf('Predicting pairwise SVM on test set...\n');

predictions_t = table('Size', [size(XTest, 1), 1], 'VariableTypes', {'categorical'});

for i = 1 : n_classifiers
    
    predictions_t{:,i} = predict(SVMModels{i}, XTest);
    
end

% transpose table
predictions_as_array = table2array(predictions_t);
oldVariableNames = predictions_t.Properties.VariableNames;
predictions_t = array2table(predictions_as_array.');
predictions_t.Properties.RowNames = oldVariableNames;

% convert to matrix
predictions = table2array(predictions_t);

YPred = [];

for i = 1 : size(XTest, 1)
    
    YPred = [YPred mode(predictions(:,i))];
    
end

YPred = YPred.';

fprintf('Accuracy on test set: %f\n', mean(YPred == YTest));


%% Fit ECOC SVM image classifier

fprintf('Doing the same for ECOC SVM...\n');

t = templateLinear();
ecoc_svm = fitcecoc(XTrain, YTrain, 'Learners', 'Linear');

YPred_ecoc = predict(ecoc_svm, XTest);

fprintf('Accuracy (ECOC) on test set: %f\n', mean(YPred_ecoc == YTest));



