%% Beispiele fuer AD

%% Ableitungen mit dem komplexen Trick
f = @(x) 2*x^3;
adsimple1(f, 2)

adsimple(@newtonSqrt, 2)

%% Zwei Variable
f = @(w) w(1)^2*w(2);
adsimple(f, [3;2])

%% Regressionsbeispiel
% Daten erzeugen
N = 100;
x = rand(N, 1);
btrue = 0.4;
wtrue = -1.5;
y = btrue + wtrue*x + 0.2 * randn(N,1);
D = [x, ones(N,1)];
wbng = linsolve(D'*D, D'*y);
fD = figure('WindowStyle', 'docked');
scatter(x,y, 'filled', 'DisplayName', 'Daten');
hold on;
plot(x, wbng(2) + wbng(1)*x, 'DisplayName', 'Fit (Norm-Gl.)');
hold off;
legend('Location','NE');


%% Gerade fitten per Gradientenabstieg
costFunc = @(wb) MSEgerade(x, y, wb);
gradFunc = @(wb) adsimple(costFunc, wb);
eta = 0.8;
nIts = 60;
wb0 = [1;1];
[wbopt, track] = gaEinfach(gradFunc, eta, wb0, nIts);
hold on;
plot(x, wbopt(1)*x + wbopt(2), ...
    'DisplayName', sprintf('Fit (Grad. Abst., #Its: %i)', nIts));
hold off;
xlabel('x'), ylabel('y');
title('Regressionsgerade mit Gradientenabstieg')


%% Animation der GA-Optimierung und Plot der Kostenfunktion
costVec = arrayfun(@(i) costFunc(track(:,i)), 1:nIts);
for i =1:nIts
    w = track(1, i);
    b = track(2, i);
    tiledlayout(2,1);
    nexttile
    scatter(x,y, 'filled');
    hold on, plot(x, w*x+b), hold off;
    nexttile;
    plot(1:nIts, costVec);
    hold on, scatter(i, costVec(i), 'r', 'filled'), hold off;
    xlabel('Iteration'), ylabel('Kostenfunktion');
    pause(0.2);

end


%% Beispiel fuer Matlabs dl-Framework
f = @(x) 2*x^3;
dlGrad(f, 2)

dlGrad(@newtonSqrt, 2)

%% Zwei Variable
f = @(w) w(1)^2*w(2);
dlGrad(f, [3;2])

%% Nochmal das Regressionsbeispiel, jetzt mit dem fl-Framework

% Modell definieren
layers = [featureInputLayer(1),
    fullyConnectedLayer(1)];
mdl = dlnetwork(layers);
summary(mdl)
% analyzeNetwork(mdl)



%% Trainingsloop
dlx = dlarray(x, 'BC');
dly = dlarray(y, 'BC');

eta = 0.1;
vel = [];    % Speichern des momentums
nEpochs = 100;
for i = 1:nEpochs
    [loss, grad] = dlfeval(@dlMSE, mdl, dlx, dly);
    fprintf('loss = %f\n', loss);
    % Stoch. GA mit momentum
    [mdl,vel] = sgdmupdate(mdl, grad, vel, eta);
end
% Gefundene Parameter
wbml = [extractdata(mdl.Learnables.Value{1}), extractdata(mdl.Learnables.Value{2})];
% Anwenden des Modells
ypredml = extractdata(forward(mdl, dlx));
hold on;
plot(x, ypredml, 'DisplayName', 'ML-Framework');
hold off;


%% Kuerzere Variante mit wrapper 'trainNetwork' (trainnet seit 24a)
mdl2 = [layers; ...
    regressionLayer];   % impliziert MSE-Kostenfunktion
options = trainingOptions('sgdm',...
    'MaxEpochs', nEpochs, ...
    'MinibatchSize', N, ...
    'InitialLearnRate', eta, ...
    'Verbose', true, ...
    'Plots', 'None');
[net, info] = trainNetwork(x, y, mdl2, options);

% Anwenden des Modells
ypredml2 = predict(net, x);
hold on;
plot(x, ypredml2, DisplayName="ML-Framework (V2)");
hold off;