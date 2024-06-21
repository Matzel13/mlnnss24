%% Die logistische DGL
% Lösen der logistischen DGL f'(x) = f(x)*(1-f(x)) mit einem NN.
% https://towardsdatascience.com/solving-differential-equations-with-neural-networks-afdcf7b8bcc4

%% Die exakte Lösung und 
R = 1;
yexakt = @(x) 1./(1+exp(-R*x));
y0 = 1/2;

nPoints = 10;
xx = linspace(0, 5, 100);
collocPoints =  linspace(0, 5, nPoints);
pl1 = plot(xx, yexakt(xx), 'DisplayName', 'Exakt');
hold on; 
pl2 = scatter(0, y0, "filled", DisplayName="Anfangswert");
pl3 = xline(collocPoints(1), 'k:', DisplayName="Kollokations-Pkte");
xline(collocPoints(2:end), 'k:');
hold off; 
xlim([xx(1), xx(end)]);
legend([pl1, pl2, pl3], 'Location', 'SE');
title('Lösung der log. DGL')
%% Jetzt das Netz
%Die Aktivierungsfunktion muss passen: sigmoid und relu klappen nicht gut, swish und tanh klappen gut
actLayer = geluLayer;
L = 12;
layers = [featureInputLayer(1), 
   fullyConnectedLayer(L)
   actLayer
   fullyConnectedLayer(L)
   actLayer
   fullyConnectedLayer(1)];
model = dlnetwork(layers);
summary(model);

%% Auswerten des untrainierten Modells
dlcolloc = dlarray(collocPoints, 'CB'); 
dly = forward(model, dlcolloc);
hold on; 
plot(collocPoints, extractdata(dly), DisplayName="Untrainiert");
hold off; 

%% Training
% Probe-Auswertung der Kostenfunktion
lossFcn = dlaccelerate(@logisticDglLoss); 
[c,g] = dlfeval(lossFcn, model, dlcolloc , y0, 1);

% Trainings-Loop:
lambda = 1; 
vel = [];
eta = 0.02;
tic
for i = 1:1500
   [loss, grad] = dlfeval(lossFcn, model, dlcolloc , y0, lambda);
   if mod(i, 100) == 0
       fprintf('%i: loss = %f\n', i, loss);
   end
   [model,vel] = sgdmupdate(model,grad,vel, eta);
end
toc
dly = forward(model, dlarray(xx, 'CB'));
hold on; 
plot(xx, extractdata(dly), DisplayName="Trainiert");
% ylim([0,1]);
hold off;
legend(Location="SE");

%% Jetzt mit variablem R!
R = 1.3684;
yexakt = @(x) 1./(1+exp(-R*x));

% erzeuge verrauschte Messdaten
rng(2);   % seed
dglDatenX = [0, sort(5*rand(1, 9))];  % 0 (d.h. die AB) soll Teil der Daten sein 
dglDatenY = yexakt(dglDatenX); 
dglDatenY(2:end) = dglDatenY(2:end) + 0.03*randn(1,9);


pl1 = plot(xx, yexakt(xx), 'Linewidth', 1, 'DisplayName', "Exakte Lösung"); % sprintf('Exakt, R=%.2f', R));
title("Lösung der logistischen DGL");
hold on; 
pl2 = xline(collocPoints(1), 'k:', DisplayName="Kollokations-Pkte");
xline(collocPoints(2:end), 'k:');

pl3 = scatter(dglDatenX, dglDatenY, "filled", DisplayName="Messpunkte");
hold off; 
legend([pl1, pl3, pl2], 'Location', 'SE');
%% Jetzt das Netz
%Die Aktivierungsfunktion muss passen: sigmoid und relu klappen nicht gut, swish und tanh klappen gut
actLayer = geluLayer;
L = 12;
layers = [featureInputLayer(1), 
   fullyConnectedLayer(L)
   actLayer
   fullyConnectedLayer(L)
   actLayer
   fullyConnectedLayer(1)];
model = dlnetwork(layers);
summary(model);

%% Auswerten des untrainierten Modells
dlcolloc = dlarray(collocPoints, 'CB'); 
dly = forward(model, dlcolloc);
hold on; 
plot(collocPoints, extractdata(dly), DisplayName="Untrainiert");
hold off; 



%% Probe-Auswertung
lossFcn = dlaccelerate(@logisticDglLossVarR); 
lambdaAB = 10; 
lambdaDat = 1; 
dlR = dlarray(R);
[c,g] = dlfeval(lossFcn, model, dlcolloc, ...
    dlarray(dglDatenX, 'CB'), dlarray(dglDatenY, 'CB'), ...
    dlR, lambdaDat, lambdaAB);

%% Jetzt ein Trainings-Loop
vel = [];
velR = []; 
dlR = dlarray(1.3);
eta = 0.01;
etaR = 0.2; 
tic
for i = 1:3000
   
   [loss, grads] = dlfeval(lossFcn, model, dlcolloc, ...
       dlarray(dglDatenX, 'CB'), dlarray(dglDatenY, 'CB'), dlR, ...
           lambdaDat, lambdaAB);
  
   
   [model,vel] = sgdmupdate(model,grads{1},vel, eta);
   [dlR,velR] = sgdmupdate(dlR,grads{2},velR, etaR);

   
   if mod(i, 50) == 0
       fprintf('%i: loss = %f, R = %.2f\n', i, loss, extractdata(dlR));
    end
end
dly = forward(model, dlarray(xx, 'CB'));
hold on; 
plot(xx, extractdata(dly), DisplayName="Trainiert");
ylim([0,1]);
hold off;
legend(Location="SE");

