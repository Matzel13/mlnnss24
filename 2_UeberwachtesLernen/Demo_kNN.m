%% kNN als Beispiel fuer einen Lernalgo

%% Trainingsdaten laden
% https://allisonhorst.github.io/palmerpenguins/
dateiName = fullfile('..', 'Daten', 'penguins_train.csv');
T = readtable(dateiName);
disp(head(T));
trainMat = [T.bill_length_mm, T.flipper_length_mm];
trainLbl = categorical(T.species);
tabulate(trainLbl);
nTrain = length(trainLbl);

%% Scatterplot
tiledlayout(2,1, 'TileSpacing','compact', 'Padding', 'compact');
nexttile;
gscatter(trainMat(:,1), trainMat(:,2), trainLbl, 'rgb');
title('Trainingsdaten');
xlabel('Schnabellänge (mm)'), ylabel('Flossenlänge (mm)');
xdim = [30,60];
ydim = [170, 240];
axis([xdim, ydim]);

%% Trainieren des knn

k = 5;
kNN = fitcknn(trainMat, trainLbl, 'NumNeighbors',k);

%% Anwenden auf die Daten, die wir haben:
trainLblPred = predict(kNN, trainMat);
nexttile;
gscatter(trainMat(:,1), trainMat(:,2), trainLblPred, 'rgb');
axis([xdim, ydim]);
title('Trainingsdaten mit kNN-Labeln');
mask = trainLblPred ~= trainLbl;

%% Fehlklassifizierungen sichtbar machen
hold on;
scatter(trainMat(mask,1), trainMat(mask, 2), 72, 'kx', 'Linewidth', 2, ...
   'DisplayName', 'Trainingsfehler');
legend off;
hold off

%% Klassifizierungsregionen
nGrid = 200;
[X,Y] = meshgrid(linspace(xdim(1), xdim(2), nGrid), ...
    linspace(ydim(1), ydim(2), nGrid));
[~, S] = predict(kNN, [X(:), Y(:)]);
S = reshape(S, [nGrid, nGrid, 3]);
hold on;
image(S, 'XData', xdim, 'YData', ydim);
alpha(0.4);   % Transparenzstufe
hold off;
axis([xdim, ydim]);





%% Berechnung Trainingsfehler
trainErr = sum(trainLblPred ~= trainLbl) / nTrain;
trainAcc = 1 - trainErr;
fprintf("Trainingsfehler: %.1f%%\n", trainErr*100);



%% Testen mit weiterem Datensatz
dateiName = fullfile('..', 'Datensaetze', 'penguins_test.csv');
Ttest = readtable(dateiName);
disp(head(Ttest));
testMat = [Ttest.bill_length_mm, Ttest.flipper_length_mm];
testLbl = categorical(Ttest.species);
nTest =  length(testLbl);
tabulate(testLbl);

%% Scatterplot
figure;
tiledlayout(2,1, 'TileSpacing','compact', 'Padding', 'compact');
nexttile;
gscatter(testMat(:,1), testMat(:,2), testLbl, 'rgb');
title('Testdaten');
axis([xdim, ydim]);

%% Anwenden von kNN
testLblPred = predict(kNN, testMat);
nexttile;
gscatter(testMat(:,1), testMat(:,2), testLblPred, 'rgb');
title('Testdaten mit kNN Labeln');
axis([xdim, ydim]);

%% Analyse der Testfehler
testErr = mean(testLblPred ~= testLbl);
testAcc = 1 - testErr;
mask = testLbl ~= testLblPred;
hold on;
scatter(testMat(mask, 1), testMat(mask, 2), 72, 'kx', 'Linewidth', 2, ...
   'DisplayName', 'Testfehler');
hold off
title(sprintf('Testdaten mit kNN Labeln, Testfehler=%.2f%%', testErr*100));
legend off;

%% Kreuzvalidierung mit den Trainingsdaten
nFolds = 5;   % mit 80% wird jeweils trainiert
nRuns = 50;
xValErr = zeros(nRuns*nFolds, 1);
xTrainErr = zeros(nRuns*nFolds, 1);
count = 1;
for r = 1:nRuns
    cv = cvpartition(nTrain, 'KFold', nFolds);
    for i = 1:nFolds
        trainMask = training(cv, i);
        kNN = fitcknn(trainMat(trainMask,:), trainLbl(trainMask), ...
            'NumNeighbors',k);
        % xval-Validierungsfehler
        pred = predict(kNN, trainMat(~trainMask,:));
        xValErr(count) = mean(pred ~= trainLbl(~trainMask));
        % xval-Trainingsfehler
        pred = predict(kNN, trainMat(trainMask,:));
        xTrainErr(count) = mean(pred ~= trainLbl(trainMask));
        count = 1 + count;
    end
end


%% Darstellung als Histogramm
figure;
histogram(xValErr, 8);
hold on; histogram(xTrainErr, 8); hold off;
legend('X-Validierungsfehler', 'X-Trainingsfehler');
title({sprintf('kNN, k= %i', k);
    sprintf('Mittlerer XVal-Fehler: %.3f%%', 100*mean(xValErr));
    sprintf('Mittlerer Trainingsfehler: %.3f%%', 100*mean(xTrainErr));});




%% knn und Spiral data
load(fullfile('..', 'Daten', 'spiral'));
xdim = [min(xySpiral(:,1)), max(xySpiral(:,1))];
ydim = [min(xySpiral(:,2)), max(xySpiral(:,2))];
gscatter(xySpiral(:,1), xySpiral(:,2), lblSpiral, 'rg', '..', 15*[1,1]);
xlim(xdim), ylim(ydim);
nTrain = size(xySpiral, 1);

%% knn anwenden
k =250;
kNN = fitcknn(xySpiral, lblSpiral, 'NumNeighbors',k);
% Klassifikationsregionen sichtbar machen
nGrid = 250;
[X,Y] = meshgrid(linspace(xdim(1), xdim(2), nGrid), ...
    linspace(xdim(1), xdim(2), nGrid));
Z = predict(kNN, [X(:), Y(:)]);
Z = reshape(Z, size(X));
Z = cat(3, 1-Z, Z, zeros(size(X)));   % b-Kanal = 0
hold on;
image(Z, 'XData', xdim, 'YData', ydim);
alpha(0.4);   % Transparenzstufe
hold off;
axis([xdim, ydim]);

%% Kreuzvalidierung
nFolds = 5;
nRuns = 5;
kVec = [1:21, 23:2:41, 45:5:175];
% Kreuzvalidierungsfehler fuer jedes k:
meanxValErr = zeros(size(kVec));
meanxTrainErr = zeros(size(kVec));
tic;
for kk=1:length(kVec)
    fprintf("Anzahl Nachbarn: %i\n", kVec(kk));
    xValErr = zeros(nRuns*nFolds, 1);
    xTrainErr = zeros(nRuns*nFolds, 1);
    count = 1;
    for r = 1:nRuns
        cv = cvpartition(nTrain, 'KFold', nFolds);
        for i = 1:nFolds
            trainMask = training(cv, i);
            kNN = fitcknn(xySpiral(trainMask,:), lblSpiral(trainMask), ...
                'NumNeighbors',kVec(kk));
            % xval-Validierungsfehler
            pred = predict(kNN, xySpiral(~trainMask,:));
            xValErr(count) = mean(pred ~= lblSpiral(~trainMask));
            % xval-Trainingsfehler
            pred = predict(kNN, xySpiral(trainMask,:));
            xTrainErr(count) = mean(pred ~= lblSpiral(trainMask));
            count = 1 + count;
        end
    end
    meanxValErr(kk) = mean(xValErr);
    meanxTrainErr(kk) = mean(xTrainErr);
 end
toc

%% Graphisch darstellen
plot(kVec, meanxValErr, 'r', kVec, meanxTrainErr, 'b', 'Linewidth', 2);
legend('XVal-Fehler', 'Trainingsfehler');
ylim([0, 0.3]);
xlabel('Anzahl Nachbarn k');
ylabel('Fehlerrate');
title('Kreuzvalidierung knn');
