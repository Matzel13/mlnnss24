%% Bayes Classfikator in 2D
% Der gleiche Datensatz, der auch schon mit dem kNN analysiert wurde

%% Trainingsdaten laden und plotten
dateiName = fullfile('..', 'Daten', 'penguins_train.csv');
T = readtable(dateiName); 
disp(head(T)); 
% Speichern als Datenmatrix und kategorielles Array
trainMat = [T.bill_length_mm, T.flipper_length_mm]; 
trainLbl = categorical(T.species);
cats = categories(trainLbl);
tabulate(trainLbl); 
% Daten plotten: 
f1 = figure;
gscatter(trainMat(:,1), trainMat(:,2), trainLbl, 'rgb'); 
title('Trainingsdaten'); 
xlabel('Schnabellänge (mm)'), ylabel('Flossenlänge (mm)'); 
xdim = [30,60]; 
ydim = [170, 240]; 
axis([xdim, ydim]); 

%% Schaetze Priors aus den Trainingsdaten
m1 = trainLbl=='Adelie';
Pri1 = mean(m1);
m2 = trainLbl=='Chinstrap';
Pri2 = mean(m2);
m3 = trainLbl=='Gentoo';
Pri3 = mean(m3);

%% Histogramme
f2 = figure; 
nBins = 20; 
t = tiledlayout(3,2);
t.TileSpacing = 'compact';
t.Padding = 'compact';
nexttile, yyaxis left;
histogram(trainMat(m1,1), nBins); 
xlim(xdim); 
title('Adelie, x'); 
nexttile, yyaxis left;
histogram(trainMat(m1,2), nBins);
xlim(ydim); 
title('Adelie, y'); 

nexttile, yyaxis left;
histogram(trainMat(m2,1), nBins); 
xlim(xdim); 
title('Chinstrap, x'); 
nexttile, yyaxis left;
histogram(trainMat(m2,2), nBins);
xlim(ydim); 
title('Chinstrapy, y'); 

nexttile, yyaxis left;
histogram(trainMat(m3,1), nBins); 
xlim(xdim); 
title('Gentoo, x'); 
nexttile, yyaxis left;
histogram(trainMat(m3,2), nBins);
xlim(ydim); 
title('Gentoo, y'); 


%% Schaetze die einzelnen Verteilungen als Gauss-Verteilung
% unnormierte Normal(=Gauss)-Verteilung
figure(f2);
gauss = @(x,m,s) 1/s * exp(-1/2*(x-m).^2/(s^2)); 
xx = linspace(xdim(1), xdim(2), 200);
yy = linspace(ydim(1), ydim(2), 200);

% Gruppe 1: Adelie
mu1 = mean(trainMat(m1,:)); 
s1 = std(trainMat(m1,:));
nexttile(1); 
yyaxis right;
hold on, 
plot(xx, gauss(xx, mu1(1), s1(1))), 
ylim([0, 1.1]); 
hold off; 
nexttile(2); 
yyaxis right;
hold on, plot(yy, gauss(yy, mu1(2), s1(2))), hold off; 

% Gruppe 2: Chinstrapy
mu2 = mean(trainMat(m2,:)); 
s2 = std(trainMat(m2,:)); 
nexttile(3); 
yyaxis right;
hold on, plot(xx, gauss(xx, mu2(1), s2(1))), hold off; 
nexttile(4); 
yyaxis right;
hold on, plot(yy, gauss(yy, mu2(2), s2(2))), hold off; 


% Gruppe 3: Gentoo
mu3 = mean(trainMat(m3,:)); 
s3 = std(trainMat(m3,:)); 
nexttile(5); 
yyaxis right;
hold on, plot(xx, gauss(xx, mu3(1), s3(1))), hold off; 
nexttile(6); 
yyaxis right;
hold on;
plot(yy, gauss(yy, mu3(2), s3(2)));
ylim([0, 1.1]); 
hold off; 



%% Posteriors berechnen und darstellen
% Naive-Bayes-Annahme: Die class-conditionals sind das Produkt der
% Verteilungen der einzelnen Merkmale!
cc1 = @(xvec) gauss(xvec(1), mu1(1), s1(1)) .* gauss(xvec(2), mu1(2), s1(2));
cc2 = @(xvec) gauss(xvec(1), mu2(1), s2(1)) .* gauss(xvec(2), mu2(2), s2(2));
cc3 = @(xvec) gauss(xvec(1), mu3(1), s3(1)) .* gauss(xvec(2), mu3(2), s3(2));


nGrid = 250; 
[X,Y] = meshgrid(linspace(xdim(1), xdim(2), nGrid), ...
    linspace(ydim(1), ydim(2), nGrid)); 
img = zeros(nGrid, nGrid, 3);
for ix = 1:nGrid
   for iy = 1:nGrid
      pos = [X(ix,iy), Y(ix,iy)]; 
      postProbs = [cc1(pos)*Pri1, cc2(pos)*Pri2, cc3(pos)*Pri3]; 
      postProbs = postProbs / sum(postProbs); % Normierung
      img(ix,iy,:) = postProbs; 
   end
end
% In den Scatter-Plot einzeichnen
figure(f1);
hold on; 
image(img, 'XData', xdim, 'YData', ydim);
alpha(0.4);   % Transparenzstufe
hold off; 

%% Die Posteriors noch mal als Surface-Plots
f3 = figure;
Cr = zeros(size(img)); 
Cr(:,:,1) = 1; 
surf(X,Y,img(:,:,1), Cr); 
shading interp; 
alpha(0.5)
hold on; 
Cg = zeros(size(img)); 
Cg(:,:,2) = 1; 
surf(X,Y,img(:,:,2), Cg); 
shading interp; 
alpha(0.5)

Cb = zeros(size(img)); 
Cb(:,:,3) = 1; 
surf(X,Y,img(:,:,3), Cb); 
shading interp; 
alpha(0.5)
hold off; 


%% Klassifizierung: Trainingsfehler
nTrain = size(trainMat, 1);
probs = zeros(nTrain,3);
for n = 1:nTrain
   probs(n,1) = cc1(trainMat(n,:)) * Pri1;
   probs(n,2) = cc2(trainMat(n,:)) * Pri2;
   probs(n,3) = cc3(trainMat(n,:)) * Pri3;
end
[~, trainLblPred] = max(probs, [], 2); 
% Umwandeln in categoriellen Vektor:
trainLblPred = categorical(trainLblPred, 1:3, categories(trainLbl));
errTrain = mean(trainLbl ~= trainLblPred);
%% Klassifizierung: Testfehler
dateiName = fullfile('..', 'Daten', 'penguins_test.csv');
Ttest = readtable(dateiName); 
testMat = [Ttest.bill_length_mm, Ttest.flipper_length_mm]; 
testLbl = categorical(Ttest.species);

nTest = size(testMat, 1);
probs = zeros(nTest,3);
for n = 1:nTest
   probs(n,1) = cc1(testMat(n,:)) * Pri1;
   probs(n,2) = cc2(testMat(n,:)) * Pri2;
   probs(n,3) = cc3(testMat(n,:)) * Pri3;
end
[~, testLblPred] = max(probs, [], 2); 
% Umwandeln in categoriellen Vektor:
testLblPred = categorical(testLblPred, 1:3, categories(testLbl));

errTest = mean(testLbl ~= testLblPred);


