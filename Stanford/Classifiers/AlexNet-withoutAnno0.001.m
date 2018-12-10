network3 = alexnet;
layers3 = network3.Layers;
layers3(23) = fullyConnectedLayer(31);
layers3(25) = classificationLayer

imds3 = imageDatastore('C:\Users\Camille\Desktop\IFT6390\Projet\Stanford\JPEG\','IncludeSubfolders', true, 'LabelSource', 'foldernames')

inputSize = layers3(1).InputSize(1:2); 

imds3.ReadFcn = @(loc)imresize(imread(loc),inputSize);

[trainingImages3, testImages3] = splitEachLabel(imds3, 0.8, 'randomize');

opts3 = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 30, 'MiniBatchSize', 64,'Plots','training-progress');

trainingImages3.ReadFcn = @(loc)imresize(imread(loc),[227,227]);
%size = trainingImages3.InputSize

myNet3 = trainNetwork(trainingImages3, layers3, opts3);

save myNet3

%Alexnet with 0.001,20,64 without annotations
%testImages3.ReadFcn = @(loc)imresize(imread(loc),[227,227]);
predictedLabels3 = classify(myNet3, testImages3);
accuracy3 = mean(predictedLabels3 == testImages3.Labels)

save testImages3
save predictedLabels3
save accuracy3