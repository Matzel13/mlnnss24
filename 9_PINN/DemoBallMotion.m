%% Demo ball motion
% https://towardsdatascience.com/physics-informed-neural-networks-pinns-an-intuitive-guide-fff138069563

%% ODE-Solver für genaue Loesung
options = odeset('Events', @myBallMotionEvent); 
g = 9.81;
mu = 0.03; 
odeFunc = @(t, zv) [zv(3); ...
    zv(4); ...
    -mu*sqrt(zv(3)^2 + zv(4)^2)*zv(3);
    -mu*sqrt(zv(3)^2 + zv(4)^2)*zv(4) - g];
zv0 = [0; 0; 20; 20];
sol = ode45(odeFunc, [0, 20], zv0, options);




%% Ein paar zufällige Messpunkte im ersten Drittel des Zeitbereichs
Tend = sol.x(end);   % Flugdauer
tt = linspace(0, Tend, 100);
zz = deval(sol, tt);
z1 = zz(1,:);
z2 = zz(2, :);
nPoints = 15; 
tmess = [0.0, sort(Tend/3*rand(1, nPoints-1))]; 
yy = deval(sol, tmess);
noiseLevel = 0.1; 
xmess = yy(1,:) + noiseLevel * randn(size(tmess));
ymess = yy(2,:) + noiseLevel * randn(size(tmess));
% keine Unsicherheit am Anfang: 
xmess(1) = yy(1,1);
ymess(1) = yy(2,1);
f1 = figure(); 
pl3 = scatter(xmess, ymess, 'filled', 'Displayname', 'Messpunkte');

%% Plot eines Fit-Polynoms der Ordnung 4
coeffs = polyfit(xmess, ymess, 4);
hold on; 
pl5 = plot(z1, polyval(coeffs, z1), 'DisplayName', 'Fit-Poly');
hold off; 


%% Plot der Loesung
hold on; 
pl1 = plot(z1, z2, 'Displayname', 'Flugbahn', 'LineWidth', 1);
pl2 = scatter(z1(end), z2(end), 'filled', 'Displayname', 'Aufprall');
hold off;
xlabel("Weite"), ylabel("Höhe"); 

% mehr coloc Punkte ueber den ganzen Bereich
nColoc = 20; 
tColoc = linspace(0, Tend, nColoc);
yyColoc = deval(sol, tColoc); 
hold on; 
pl4 = xline(yyColoc(1,1), 'k:', DisplayName="Kollokations-Pkte");
xline(yyColoc(1,2:end), 'k:');
hold off; 
title(sprintf("Flugbahn: Aufprall nach %.2f m", z1(end)));
legend([pl1, pl5, pl3, pl4], "Location", "S");
ylim([0, 12]);


%% PINN-Berechnung der Flugbahn
actLayer = geluLayer;
L = 32;
layers = [featureInputLayer(1)
   fullyConnectedLayer(L) 
   actLayer
   fullyConnectedLayer(L) 
   actLayer
   fullyConnectedLayer(2)]; 
mdl = dlnetwork(layers);
summary(mdl);

%% Probe-Auswertung
dltColoc = dlarray(tColoc, 'CB');
dltmess = dlarray(tmess, 'CB');
dlzmess = dlarray([xmess; ymess], 'CB');

lossFcn = dlaccelerate(@ballMotionLoss);
dlmu = dlarray(2*mu);
[c,g] = dlfeval(lossFcn, mdl, dltmess, dlzmess, dltColoc, dlmu);

%% Training Loop
nRuns = 5000;
lossVec = zeros(nRuns, 1);
muVec = zeros(nRuns, 1);

eta = 0.002; 
etamu = 0.0003;
averageSqGrad = [];
averageSqGradMu = [];
learnRateDropPeriod = 2000;
learnRateDropFactor = 0.75;
tic
for i = 1:nRuns
    if mod(i,learnRateDropPeriod) == 0
       eta = eta * learnRateDropFactor;
       % etamu = etamu * learnRateDropFactor;
       fprintf("Learnrate=%f\n", eta);
    end

    if mod(i, 100) == 0
       fprintf('%i: loss = %f, mu = %f\n', i, loss, extractdata(dlmu));
    end
   [loss, grads] = dlfeval(lossFcn, mdl, dltmess, dlzmess, dltColoc,  dlmu);
   [mdl,averageSqGrad] = rmspropupdate(mdl,grads{1},averageSqGrad, eta);
   [dlmu,averageSqGradMu] = rmspropupdate(dlmu, grads{2},averageSqGradMu, etamu);
   
   lossVec(i) = loss; 
   muVec(i) = extractdata(dlmu);
end
toc


%% Plotten der Modellvorhersage
zpred = extractdata(forward(mdl, dltmess));
zpredTotal = extractdata(forward(mdl, dlarray(tt, 'CB')));
extractdata(forward(mdl, dlarray(0, 'CB')));

hold on; 
figure(f1);
scatter(zpred(1,:), zpred(2,:), 'filled', DisplayName="NN: Messpunkte");
plot(zpredTotal(1,:), zpredTotal(2,:), DisplayName="NN: gesamt");
hold off;
legend(Location="S");
%% Optimierungsverlauf
f2 = figure; 
tiledlayout(2,1);
nexttile;
plot(1:nRuns,  lossVec, '.'); 
title("Kostenfunktion")
nexttile; 
plot(1:nRuns, muVec); 
yline(mu);
legend('Schätzung NN', 'Exakt');
title("Der Parameter mu");
xlabel("Iterationen")