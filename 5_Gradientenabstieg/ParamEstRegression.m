 %% Regression per Gradientenabstieg
 
%% Bsp 1: Eine Regressionsgerade 
% Daten erzeugen
x = rand(100, 1); 
btrue = 0.4; 
wtrue = -1.5; 
y = btrue + wtrue*x + 0.2 * randn(size(x));
D = [x, ones(size(x))]; 
wbng = linsolve(D'*D, D'*y); 
fD = figure('WindowStyle', 'docked'); 
scatter(x,y, 'filled', 'DisplayName', 'Daten'); 
hold on;
plot(x, wbng(2) + wbng(1)*x, 'DisplayName', 'Fit (Norm-Gl.)'); 
hold off;
legend('Location','NE');

%% Gradientenabstieg anwenden
MSEGradFunc = @(wb) MSEgeradeGrad(x,y,wb);
eta = 0.8;
nIts = 60; 
wb0 = [1;1];   
[wbopt, track] = gaEinfach(MSEGradFunc, eta, wb0, nIts);

hold on; 
plot(x, wbopt(2) + wbopt(1)*x, ...
   'DisplayName', sprintf('Fit (Grad. Abst., #Its: %i)', nIts));
hold off;
xlabel('x'), ylabel('y'); 
title('Regressionsgerade mit Gradientenabstieg')

%% Animation der Iterationen
for n = 1:nIts
   w = track(1,n);
   b = track(2,n); 
 
   scatter(x,y, 'filled'); 
   ylim([-2,2]);
   hold on, plot(x, w*x+b), hold off; 
   title(sprintf('Iteration Nummer %i', n)); 
   drawnow, pause(0.1);
end



%% Die Energielandschaft
MSEFunc = @(wb) MSEgerade(x,y,wb);
[W,B] = meshgrid(linspace(-4,2, 100)); 
MSE = zeros(size(B));
for r=1:size(B,1)
   for c=1:size(B,2)
       wb = [W(r,c); B(r,c)];
       MSE(r,c) = MSEFunc(wb);
   end
end
figure('WindowStyle', 'docked'); 
surf(W,B, MSE+1); 
xlabel('b'), ylabel('w'), zlabel('MSE'); 
shading interp;  light, lighting gouraud, material dull
alpha(0.5)
colorbar; 
hold on; 
contour(W,B, MSE, [0:0.2:1, 1.5:0.5:5], 'LineWidth', 2); 
xlabel('b'), ylabel('w'); 
plot([wb0(1), track(1,:)], [wb0(2), track(2,:)], 'r'); 
plot(wbopt(1), wbopt(2), 'or'); 
plot(wtrue, btrue, 'og'); 
plot(wbng(1), wbng(2), 'xg'); 
hold off; 


%% Bsp 2: Fit einer abklingenden Schwingung
% Das ist ein nichtlineares Modell. Die Energielandschaft ist deutlich
% komplexer

rng(0);   % macht die Simulation reproduzierbar
t = sort(rand(100,1)); 
w1true = -0.4; 
w2true = 5; 
y = exp(w1true*t) .*sin(w2true*t) + 0.1*randn(size(t)); 
figD = figure('Name', 'Daten', 'WindowStyle', 'docked'); 
scatter(t,y, 'filled'); 
hold on; 
plot(t, exp(w1true*t) .*sin(w2true*t), 'k'); 
hold off; 
xlabel('t','Interpreter', 'latex', 'Fontsize', 14); 
ylabel('y', 'Interpreter', 'latex', 'Fontsize', 14); 
legend('Daten', 'richtige Parameter');
% MSE-Wert fuer die echten Parameter
yfit = exp(w1true*t) .* sin(w2true*t); 
MSEmin = 1/2*mean( (y-yfit).^2);
title('Schwingung mit Rauschen', ...
   sprintf('MSE=%.3f', MSEmin), 'Interpreter', 'latex', 'Fontsize', 14); 


%% Die Energielandschaft
MSEFunc = @(w) MSEschwingung(t, y, w);
[W1, W2] = meshgrid(linspace(-4,4, 100), linspace(-2,7, 100));
MSE = zeros(size(W1));
for r=1:size(W1,1)
   for c=1:size(W1,2)
       w = [W1(r,c); W2(r, c)];
       MSE(r,c) = MSEFunc(w);
   end
end

% Darstellung in 3D und in 2D als Contour-Plot
f3D = figure('Name', 'MSE-3D', 'WindowStyle', 'docked'); 
MSEcut = MSE; 
MSEcut(MSE>2) = 2.5;   % Abschneiden zur besseren Darstellung der Farben
surf(W1, W2, MSEcut); 
hold on; 
contour3(W1, W2, MSEcut); 
plot3(w1true, w2true, MSEmin, 'g.', 'MarkerSize', 14); 
hold off; 
xlabel('w_1'), ylabel('w_2'), zlabel('MSE'); 
shading interp;  light, lighting gouraud, material dull
alpha(0.8)
colorbar; 
% 2D
f2D = figure('Name', 'MSE-Contour', 'WindowStyle', 'docked');  
contour(W1, W2, MSE, [0:0.01:0.1, 0.125:0.025: 0.275, 0.3:0.01:0.4]); 
xlabel('w_1'), ylabel('w_2'); 
hold on; 
plot(w1true, w2true, 'go'); 
hold off;
axis equal; 


%% Gradientenabstieg vorbereiten 
MSEGradFunc = @(w) MSEschwingungGrad(t,y, w);
eta =  0.4;
nIts = 800; 
% Anfangswerte
w0t = [-1;1];   %% hiermit geht's ins richtige Minimum
w0f = [-1;0];   %% hiermit geht's ins falsche Minimum
% in die Plots eintrage
figure(f3D); 
hold on;
plot3(w0t(1), w0t(2), MSEFunc(w0t), 'kx', 'Linewidth', 2);
plot3(w0f(1), w0f(2), MSEFunc(w0f), 'kx', 'Linewidth', 2);
hold off;

figure(f2D); 
hold on;
plot(w0t(1), w0t(2), 'ko');
plot(w0f(1), w0f(2), 'ko');
hold off;


%% Gradientenabstieg durchfuehren
[wopt_t, wtrack_t] = gaEinfach(MSEGradFunc, eta, w0t, nIts);
[wopt_f, wtrack_f] = gaEinfach(MSEGradFunc, eta, w0f, nIts);
% Auswerten der Kostenfunktion auf der Abstiegsspur
MSEtrack_t = zeros(nIts, 1); 
MSEtrack_f = zeros(nIts, 1); 
for it=1:nIts
    MSEtrack_t(it) = MSEFunc(wtrack_t(:,it)); 
    MSEtrack_f(it) = MSEFunc(wtrack_f(:,it)); 
end
% Eintragen in den Plot
figure(f3D); 
hold on;
plot3(wtrack_t(1,:), wtrack_t(2,:), MSEtrack_t, 'bx:', 'Linewidth', 1);
plot3(wtrack_f(1,:), wtrack_f(2,:), MSEtrack_f, 'rx:', 'Linewidth', 1);
hold off;

figure(f2D); 
hold on; 
plot(wtrack_t(1,:), wtrack_t(2,:), 'bx:', 'Linewidth', 1);
plot(wtrack_f(1,:), wtrack_f(2,:), 'rx:', 'Linewidth', 1);
plot(wopt_f(1), wopt_f(2), 'kx'); 
hold off; 

%% Kostenfunktion
figK = figure('Name', 'MSE', 'WindowStyle', 'docked');
plot(1:(nIts), MSEtrack_t, 'b', 'DisplayName', 'guter Startpunkt')
hold on; 
plot(1:(nIts), MSEtrack_f, 'r', 'DisplayName', 'schlechter Startpunkt'); 
hold off; 
legend; 
xlabel('Anz. It'); ylabel('MSE');
title('Die Kostenfuntion'); 
ylim([0, 1]); 

%% Die Modellfunktionen
 
figure(figD);
yt = exp(wopt_t(1)*t) .* sin(wopt_t(2)*t);
yf = exp(wopt_f(1)*t) .* sin(wopt_f(2)*t);
hold on; 
plot(t, yt, 'b', 'DisplayName', 'guter Startpunkt'); 
plot(t, yf, 'r',  'DisplayName', 'schlechter Startpunkt'); 
hold off; 

%% Stochastischer Gradientenabstieg
MSEBatchGradFunc = @(w, m) MSEschwingungGradBatch(t,y,w,m);

nEpochs = 20;
mbSize = 10; 
eta = 5;
[woptsga_t, wtracksga_t] = sgaEinfach(MSEBatchGradFunc, eta, w0t, length(t), nEpochs, mbSize); 
[woptsga_f, wtracksga_f] = sgaEinfach(MSEBatchGradFunc, eta, w0f, length(t), nEpochs, mbSize); 

figure(f2D); 
hold on; 
plot(wtracksga_t(1,:), wtracksga_t(2,:), 'v:', 'MarkerSize', 3); 
plot(wtracksga_f(1,:), wtracksga_f(2,:), 'v:', 'MarkerSize', 3); 
hold off; 

%% Eintragen in die anderen Plots
nItsSga = size(wtracksga_t, 2);
MSEtrackSga_t = zeros(nItsSga, 1); 
MSEtrackSga_f = zeros(nItsSga, 1); 
for it=1:nItsSga
   MSEtrackSga_t(it) = MSEFunc(wtracksga_t(:,it));
   MSEtrackSga_f(it) = MSEFunc(wtracksga_f(:,it));
end

figure(f3D); 
hold on;
plot3(wtracksga_t(1,:), wtracksga_t(2,:), MSEtrackSga_t, 'c', 'Linewidth', 2);
plot3(wtracksga_f(1,:), wtracksga_f(2,:), MSEtrackSga_f, 'm', 'Linewidth', 2);
hold off;

figure(figK);
hold on; 
plot(1:nItsSga, MSEtrackSga_t, 'c', 'DisplayName', 'guter Startpunkt (sga)'); 
plot(1:nItsSga, MSEtrackSga_f, 'm', 'DisplayName', 'schlechter Startpunkt (sga)'); 
hold off;
xlim([1, nIts]); 
%%
figure(figD); 
yt2 = exp(woptsga_t(1)*t) .* sin(woptsga_t(2)*t);
yf2 = exp(woptsga_f(1)*t) .* sin(woptsga_f(2)*t);
hold on; 
plot(t, yt2, 'c', 'DisplayName', 'guter Startpunkt (SGA)'); 
plot(t, yf2, 'm', 'DisplayName', 'schlechter Startpunkt (SGA)'); 
hold off; 

