%% Vollverbundene Neuronale Netze

%% Bsp 1: 2 Gruppen in 1D
load(fullfile('..', 'Daten', 'Data1D2Gr_1.mat')); 
fD = figure('WindowStyle', 'docked'); 
gscatter(xTrain, double(trainLbl)-1, trainLbl, 'rg','ox'); 
ylim([-0.1, 1.1]); 
xlabel('x'); 
legend; 

testLbl = categorical(testLbl);
trainLbl = categorical(trainLbl);
% Groessen-Parameter

[N, P] = size(xTrain); 
C = 2;


%% 1.1) Definition des Netzes und Konfiguration des Trainings

H = 20; 
layers = [featureInputLayer(P)
   fullyConnectedLayer(H)
   reluLayer
   fullyConnectedLayer(C)
   softmaxLayer
   classificationLayer];
summary(dlnetwork(layers(1:end-1)))

options = trainingOptions('rmsprop',...
   'ValidationData', {xTest, categorical(testLbl)}, ...
   'MaxEpochs', 1000, ...
   'MinibatchSize', N, ...
   'Verbose', false, ...
   'Plots', 'None'); 
 
%% 1.2 Training des Netzes und Berechnung der Fehler
[net, info] = trainNetwork(xTrain, trainLbl, layers, options);

trainLblPredML = classify(net, xTrain); 
trainErrML = sum(trainLbl ~= trainLblPredML)/length(trainLbl);

testLblPredML = classify(net, xTest); 
testErrML = sum(testLbl ~= testLblPredML)/length(testLbl); 


%% 1.3 Darstellung der Klassifizierungswahrscheinlichkeiten
figure(fD); 
xx = linspace(-4,4)';

[~, pMat] = classify(net, xx);
hold on; 
plot(xx, pMat(:,1), 'r', 'DisplayName', 'P_1(x)');
plot(xx, pMat(:,2), 'g', 'DisplayName', 'P_2(x)');
hold off; 
ylabel('Prob'); 
title('Daten und Wahrscheinlichkeiten'); 


%% Bsp 2: Xor-Daten
% Daten laden und plotten
load(fullfile('..', 'Daten', 'Xor2D.mat'));
xdim = 1.1*[min(X(:,1)), max(X(:,1))]; 
ydim = 1.1*[min(X(:,2)), max(X(:,2))];
fD = figure('WindowStyle', 'docked'); 
gscatter(X(:,1), X(:,2), lbl, 'rg', '..', 15*[1,1]);
title('Trainingsdaten'); 
legend off; 
axis([xdim, ydim]); 

[N, P] = size(X);
C = length(categories(lbl));   % Anzahl Klassen



%% 2.1) Definition des Netzes und Konfiguration des Trainings
H = 5; 
layers = [featureInputLayer(P)
   fullyConnectedLayer(H)
   reluLayer
   fullyConnectedLayer(C)
   softmaxLayer
   classificationLayer];
summary(dlnetwork(layers(1:end-1)))

options = trainingOptions('rmsprop',...
   'ValidationData', {X, lbl}, ...
   'MaxEpochs', 1000, ...
   'MinibatchSize', N, ...
   'Verbose', false, ...
   'L2Regularization', 0.01, ...
   'Plots', 'training-progress'); 
net = trainNetwork(X, lbl, layers, options);
lblPred = classify(net, X);
err = sum(lblPred ~= lbl) / length(lbl); 

%% 2.2: Klassifizierungsregionen
nGrid = 100; 
[Xg,Yg] = meshgrid(linspace(xdim(1), xdim(2), nGrid), ...
   linspace(ydim(1), ydim(2), nGrid));
Dgrid = [Xg(:), Yg(:)];
[~, pGrid]  = classify(net, Dgrid); 
img = zeros([size(Xg), 3]);
img(:,:,1) = reshape(pGrid(:,1), size(Xg));
img(:,:,2) = 1 - img(:,:,1);
% Eintragen in den Scatterplot:
hold on;
image(img, 'XData', xdim, 'YData', ydim);
% Trennlinien
thres = 0.5;
contour(Xg,Yg,img(:,:,1), thres*[1,1], 'k', 'Linewidth', 2, 'DisplayName', 'Trennlinie'); 
alpha(0.5);  % Transparenz des Hintergrundes 
hold off; 
axis([xdim, ydim]);




%% Bsp 3: MNIST-Ziffern, 6 Klassen

load(fullfile('..', 'Daten', 'mnist6Klassen.mat')); 

imSize = 28; 
nImgsTrain = length(trainLbl); 
nImgsTest = length(testLbl); 

%% 3.1: Zufällig ein paar Bilder auswählen und zeigen
idx = randperm(nImgsTrain, 20);
t = tiledlayout(4,5, 'TileSpacing','compact', 'Padding', 'compact'); 
title(t, 'Ein paar Trainingsbilder');
for i = 1:numel(idx)
    nexttile;
    imshow(trainImgs(:,:,idx(i)));
    title(sprintf('%s', trainLbl(idx(i))));
end

%% 3.2: Datenmatrizen erstellen
% Ein Bild ist eine Zeile. Dabei werden die Pixel alle
% in einer Zeile hintereinandergehängt
trainMat = zeros(nImgsTrain, imSize*imSize); 
for ii = 1:nImgsTrain
    img = trainImgs(:,:,ii);
    trainMat(ii, :) = img(:);
end
testMat = zeros(nImgsTest, imSize*imSize); 
for ii = 1:nImgsTest
    img = testImgs(:,:,ii);
    testMat(ii, :) = img(:);
end
P = size(trainMat,2);    % Anzahl Parameter = Anz Komp des w-Vektors
C = length(categories(trainLbl));   % Anzahl Klassen


% Validierungsdatensatz abspalten
mask = rand(nImgsTrain, 1) > 0.8; 
valMat = trainMat(mask, :);
valLbl = trainLbl(mask);
trainMat = trainMat(~mask, :);
trainLbl = trainLbl(~mask);

%% 3.3: Definition des Netzes und Konfiguration des Trainings

H = 100; 
layers = [featureInputLayer(P)
   fullyConnectedLayer(H)
   reluLayer
   fullyConnectedLayer(C)
   softmaxLayer
   classificationLayer];

options = trainingOptions('rmsprop',...
   'MiniBatchSize', 128,...
   'MaxEpochs', 20, ...
   'Verbose', false, ...
   'ValidationData', {valMat, valLbl},...
   'Plots', 'Training-Progress');

%% 3.4: Training des Netzes und Berechnung der Fehler
[net, info] = trainNetwork(trainMat, trainLbl, layers, options);

% Trainings- und Testfehler
trainLblPred = classify(net, trainMat); 
trainErr = mean(trainLbl ~= trainLblPred);
valLblPred = classify(net, valMat); 
valErr = mean(valLbl ~= valLblPred);

testLblPred = classify(net, testMat); 
testErr = sum(testLbl ~= testLblPred)/length(testLbl); 

% Confusion-Matrix
cmTest = confusionmat(testLbl, testLblPred);

cc = confusionchart(cmTest, categories(testLbl));
cc.ColumnSummary = 'column-normalized'; 
cc.RowSummary = 'row-normalized';
cc.Title = sprintf('MNIST (0-5) Confusion Matrix, Err-Rate=%.1f%%', 100*testErr);



%% 3.5: Ein paar Testbilder mit vorhergesagten Labels zeigen

% idxFehl = find(testLblPred ~= testLbl);
% idx = idxFehl(randperm(length(idxFehl), 20));
idx = randperm(nImgsTest, 20);
t = tiledlayout(4,5, 'TileSpacing','compact', 'Padding', 'compact'); 
title(t, 'Ein paar Testbilder');
for i = 1:numel(idx)
    nexttile;
    imshow(testImgs(:,:,idx(i)));
    title(sprintf('%s (vorh. %s)', testLbl(idx(i)), testLblPred(idx(i))));
end


%% Bsp 4: Winkelerkennung von gedrehten Ziffern
% Daten laden und ein paar Bilder darstellen
filename = fullfile('..', 'Daten', 'GedrehteZweien.mat');
load(filename); 
nTrain = numel(YTrain);
nTest = numel(YTest); 
imSize = size(XTrain, 1:2); 

%% 4.1: Zufällig ein paar Bilder auswählen und zeigen
idx = randperm(nTrain,20);
t = tiledlayout(4,5, 'TileSpacing','compact', 'Padding', 'compact'); 
title(t, 'Ein paar Trainingsbilder');
for i = 1:numel(idx)
    nexttile;
    imshow(XTrain(:,:,:,idx(i)));
    title(sprintf('%i°', YTrain(idx(i))));
end

%% 4.2: Datenmatrizen erstellen
P = imSize(1)*imSize(2);   % Anzahl Merkmale
% Trainingsdaten
DTrain = zeros(nTrain,P); 
for n = 1:nTrain
   img = XTrain(:,:,1,n); 
   DTrain(n,:) = img(:); 
end

% Testdaten
DTest = zeros(nTest,P); 
for n = 1:nTest
   img = XTest(:,:,1,n); 
   DTest(n,:) = img(:); 
end

%% 4.3: Definition des Netzes und Konfiguration des Trainings

H = 400; 
layers = [featureInputLayer(P)
   fullyConnectedLayer(H)
   reluLayer
   fullyConnectedLayer(H/2)
   reluLayer
   fullyConnectedLayer(1)
   regressionLayer];

options = trainingOptions('rmsprop',...
   'MiniBatchSize', 500,...
   'MaxEpochs', 800, ...
   'Verbose', false, ...
   'ValidationData', {DTest, YTest},...
   'Plots', 'Training-Progress');

%% 4.4: Netz trainieren und anwenden auf Trainings- und Testdaten
[net, info] = trainNetwork(DTrain, YTrain, layers, options);

t = tiledlayout(2,1, 'TileSpacing','compact', 'Padding', 'compact');
nexttile(1); 
YTrainPred = predict(net, DTrain); 
scatter(YTrain, YTrainPred);
xlabel('Winkel: Gemessen'), ylabel('Winkel: Vorhersage'); 
RMSETrain = sqrt(1/nTrain * sum( (YTrain - YTrainPred).^2 )); 
title('Trainingsdaten', sprintf('RMSE=%.2f', RMSETrain)); 

% Vorhersage auf Testdaten:
YTestPred = predict(net, DTest); 
nexttile(2);
scatter(YTest, YTestPred); 
xlabel('Winkel: Gemessen'), ylabel('Winkel: Vorhersage'); 
RMSETest = sqrt(1/nTest * sum( (YTest - YTestPred).^2 ));
title('Testdaten', sprintf('RMSE=%.2f', RMSETest)); 

%% 4.5: Ein paar Testbilder ansehen
idx = randperm(nTrain,20);
% Sortiert nach den größten Abweichungen
% d = abs(YTest - YTestPred); 
% [~, idx] = sort(d, 'ascend');
% idx = idx(1:20);

t = tiledlayout(4,5, 'TileSpacing','compact', 'Padding', 'compact'); 
for i = 1:numel(idx)
    nexttile;
    imshow(XTest(:,:,:,idx(i)));
    title(sprintf('%i° / %i°', YTest(idx(i)), round(YTestPred(idx(i)))));
end
title(t, 'Ein paar Testbilder', 'Echter Winkel / Vorhersagewinkel');

