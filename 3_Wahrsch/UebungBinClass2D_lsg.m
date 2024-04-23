% Uebungsaufgabe Kapitel 3

%% Daten laden und plotten
dateiName = fullfile('..', 'Daten', 'binclass2D.csv');
T = readtable(dateiName); 
disp(head(T)); 
% Speichern als Datenmatrix und kategorielles Array
trainMat = [T.x, T.y]; 
trainLbl = categorical(T.Klasse);
tabulate(trainLbl); 
f1 = figure; 
gscatter(trainMat(:,1), trainMat(:,2), trainLbl,'filled'); 
title('Trainingsdaten'); 
maxCoord = 4; 
axis(maxCoord*[-1, 1, -1, 1]);

%% Histogramme
figure; 
t = tiledlayout(2,2);
nexttile;
histogram(trainMat(trainLbl=='0',1));
title("x-Koordinate, Gruppe 0");
nexttile;
histogram(trainMat(trainLbl=='0',2));
title("y-Koordinate, Gruppe 0");
nexttile;
histogram(trainMat(trainLbl=='1',1));
title("x-Koordinate, Gruppe 1");
nexttile;
histogram(trainMat(trainLbl=='1',2));
title("y-Koordinate, Gruppe 1");


%% Trainieren:
Mdl = fitcnb(trainMat, trainLbl, ...
    'DistributionNames','kernel', 'Width',[0.1, 0.1]); 
% per default werden Normalverteilungen angenommen
[trainLblPred, trainPost] = predict(Mdl, trainMat); 
trainErr = mean(trainLblPred ~= trainLbl);

%% Klassifizierungsregionen
nGrid = 250; 
[X,Y] = meshgrid(linspace(-maxCoord, maxCoord, nGrid), ...
    linspace(-maxCoord, maxCoord, nGrid)); 
mat = [X(:) Y(:)];
[~, probs] = predict(Mdl, mat);
img = zeros(nGrid, nGrid, 3);
img(:,:,1) = reshape(probs(:,2), nGrid, nGrid);
img(:,:,3) = reshape(probs(:,1), nGrid, nGrid);

% In den Scatter-Plot einzeichnen
figure(f1);
hold on; 
image(img, 'XData', maxCoord*[-1, 1], ...
    'YData', maxCoord*[-1,1]);
alpha(0.4);   % Transparenzstufe
hold off; 

%% Testfehler
% Daten laden und plotten
dateiName = fullfile('..', 'Daten', 'binclass2D_test.csv');
T = readtable(dateiName); 
disp(head(T)); 
% Speichern als Datenmatrix und kategorielles Array
testMat = [T.x, T.y]; 
testLbl = categorical(T.Klasse);
tabulate(testLbl); 
figure; 
gscatter(testMat(:,1), testMat(:,2), testLbl,'filled'); 
title('Testdaten'); 
axis(maxCoord*[-1, 1, -1, 1]);

%% Testfehler
[testLblPred, testPost] = predict(Mdl, testMat); 
testErr = mean(testLblPred ~= testLbl);

