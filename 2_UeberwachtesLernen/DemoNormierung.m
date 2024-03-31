%% Beispiel fuer Normierung der Daten

load(fullfile('..', 'Daten', 'DemoNormierungDaten.mat'));
gscatter(D(:,1), D(:,2), lbl);
%%
kNN = fitcknn(D, lbl, 'NumNeighbors',11);
lblNeu = predict(kNN, D);
err = sum(lblNeu ~= lbl) / length(lbl);
disp(err);
title(sprintf('Originaldaten: Fehlerrate = %.1f%%', 100*err));

%% Skaliere die Merkmale!
m = mean(D);
sigma = std(D);
Ds = (D-m)./sigma;
figure;
gscatter(Ds(:,1), Ds(:,2), lbl);

kNN = fitcknn(Ds, lbl, 'NumNeighbors',11);
lblNeu = predict(kNN, Ds);
err2 = sum(lblNeu ~= lbl) / length(lbl);
disp(err2);
title(sprintf('Normierte Daten: Fehlerrate = %.1f%%', 100*err2));

%% Nochmal der Pinguin-Datensatz
% https://allisonhorst.github.io/palmerpenguins/
dateiName = fullfile('..', 'Daten', 'penguins_train.csv');
T = readtable(dateiName);
disp(head(T));
trainMat = [T.bill_length_mm, T.flipper_length_mm];
trainLbl = categorical(T.species);
tabulate(trainLbl);
nTrain = length(trainLbl);
% Scatterplot:
tiledlayout(2,1, 'TileSpacing','compact', 'Padding', 'compact');
nexttile;
gscatter(trainMat(:,1), trainMat(:,2), trainLbl, 'rgb');
title('Trainingsdaten (unnormiert)');
xlabel('Schnabellänge (mm)'), ylabel('Flossenlänge (mm)');
xdim = [30,60];
ydim = [170, 240];
axis([xdim, ydim]);

kNN = fitcknn(trainMat, trainLbl, 'NumNeighbors',5);
% Anwenden auf die Daten, die wir haben:
trainLblPred = predict(kNN, trainMat);
nexttile;
gscatter(trainMat(:,1), trainMat(:,2), trainLblPred, 'rgb');
axis([xdim, ydim]);
title('Trainingsdaten (unnormiert) mit kNN-Labeln');
mask = trainLblPred ~= trainLbl;
% Fehlklassifizierungen sichtbar machen
hold on;
scatter(trainMat(mask,1), trainMat(mask, 2), 72, 'kx', 'Linewidth', 2, ...
   'DisplayName', 'Trainingsfehler');
legend off;
hold off

%% Das ganze nochmal normierten Daten
m = mean(trainMat);
sigma = std(trainMat);
trainMats = (trainMat-m)./sigma;
figure;
tiledlayout(2,1, 'TileSpacing','compact', 'Padding', 'compact');
nexttile;
gscatter(trainMats(:,1), trainMats(:,2), trainLbl, 'rgb');
title('Trainingsdaten (normiert)');
xlabel('Schnabellänge (mm)'), ylabel('Flossenlänge (mm)');


kNN = fitcknn(trainMats, trainLbl, 'NumNeighbors',5);
% Anwenden auf die Daten, die wir haben:
trainLblPred = predict(kNN, trainMats);
nexttile;
gscatter(trainMats(:,1), trainMats(:,2), trainLblPred, 'rgb');
title('Trainingsdaten (normiert) mit kNN-Labeln');
mask = trainLblPred ~= trainLbl;
% Fehlklassifizierungen sichtbar machen
hold on;
scatter(trainMats(mask,1), trainMats(mask, 2), 72, 'kx', 'Linewidth', 2, ...
   'DisplayName', 'Trainingsfehler');
legend off;
hold off
trainErr = sum(trainLblPred ~= trainLbl) / nTrain;
trainAcc = 1 - trainErr;
fprintf("Trainingsfehler (normiert): %.1f%%\n", trainErr*100);
