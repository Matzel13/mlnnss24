%% Mal wieder die Palmer Penguins

%% Laden und plotten der Daten
fileName = fullfile('..', 'Daten', 'penguins_train.csv');
T = readtable(fileName);
disp(head(T));
trainMat = [T.bill_length_mm, T.flipper_length_mm];
trainLbl = categorical(T.species);
tabulate(trainLbl);
nTrain = length(trainLbl);

m = mean(trainMat);
sigma = std(trainMat);
trainMat = (trainMat-m)./sigma;



%% Netzwerk mit einer inneren Schicht
[N, P] = size(trainMat);
H = 5;
C = 3;

layers = [featureInputLayer(P)
    fullyConnectedLayer(H)
    reluLayer
    fullyConnectedLayer(C)
    softmaxLayer
    classificationLayer];

%%
lambda = 0.0;
options = trainingOptions('rmsprop',...
    'InitialLearnRate', 0.001, ...
    'ValidationData', {trainMat, trainLbl}, ...
    'MaxEpochs', 500, ...
    'MinibatchSize', N, ...
    'L2Regularization', lambda, ...
    'Verbose', true, ...
    'OutputNetwork', 'best-validation-loss',...
    'Plots', 'none');
% Training:
tic
net = trainNetwork(trainMat, trainLbl, layers, options);
toc
lblPred = classify(net, trainMat);
err1 = mean(lblPred ~= trainLbl);
disp(err1);
