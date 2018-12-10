imds = imageDatastore('C:\Users\Camille\Desktop\IFT6390\Projet\Stanford\stanford_data_set\visual_annotation\1', 'IncludeSubfolders', true, 'LabelSource', 'foldernames')

[trainingImages, testImages] = splitEachLabel(imds, 0.8, 'randomize');

network = alexnet;
layers = network.Layers;
layers(23) = fullyConnectedLayer(20);
layers(25) = classificationLayer

opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 30, 'MiniBatchSize', 64,'Plots','training-progress');

trainingImages.ReadFcn = @(loc)imresize(imread(loc),[227,227]);

myNet = trainNetwork(trainingImages, layers, opts);

save myNet

%alexnet Avec annotation 0.001
testImages.ReadFcn = @(loc)imresize(imread(loc),[227,227]);
predictedLabels = classify(myNet, testImages);
accuracy1 = mean(predictedLabels == testImages.Labels)

save testImages
save predictedLabels
save accuracy1


