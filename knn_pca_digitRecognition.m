clear
close all

dataset = load('MNIST_partitioned')

trainImages = (dataset.trainImages1000);
testImages = (dataset.testImages100);
trainLabels = (dataset.trainLabels1000);
testLabels = (dataset.testLabels100);

reshapedTrainImages = double(reshape(trainImages, [784,10000]))';
reshapedTestImages = double(reshape(testImages, [784,1000]))';

mu = mean(reshapedTrainImages,1);
reshapedTrainImages = bsxfun(@minus,reshapedTrainImages,mu);

coeff = pca(reshapedTrainImages);
Ncomponents = 120;
eigenfaces = coeff(:,1:Ncomponents);

trainFeatures = eigenfaces'*reshapedTrainImages';

reshapedTestImages = bsxfun(@minus,reshapedTestImages,mu);

testFeatures = eigenfaces'*reshapedTestImages';

md1 = fitcknn(trainFeatures',trainLabels);

labels = predict(md1,testFeatures');

correctRec = find(testLabels == labels');
correctLabels = labels(correctRec);

falseRec = find(testLabels ~= labels');
falseLabels = labels(falseRec);

result = length(correctRec)/length(testLabels)*100;
fprintf('The recognition rate is: %0.3f \n', result);