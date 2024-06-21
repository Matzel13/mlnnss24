%% Spiral-Daten
load(fullfile('..', 'Daten', 'spiral'));
xdim = [min(xySpiral(:,1)), max(xySpiral(:,1))];
ydim = [min(xySpiral(:,2)), max(xySpiral(:,2))];
gscatter(xySpiral(:,1), xySpiral(:,2), lblSpiral, 'rg', '..', 15*[1,1]);
xlim(xdim), ylim(ydim);
lblSpiral = categorical(lblSpiral);
lblMat = onehotencode(lblSpiral', 1);
[N, P] = size(xySpiral);
C = 2;



%% Ein tiefes, schmales Netz
H = 25;
block = [fullyConnectedLayer(H)
   reluLayer
   batchNormalizationLayer];

layers = [featureInputLayer(P)
   block
   block
   block
   block
   block
   block
   block
   block
   block
   block
   fullyConnectedLayer(C)
   softmaxLayer];
net = dlnetwork(layers);



%% Custom Training-Loop
X = dlarray(single(xySpiral'),"CB");
% X = gpuArray(X);
numEpochs = 1000;
lossVec = zeros(1,numEpochs);
averageSqGrad = [];  % fuer rmsprop
eta = 0.001;
%% Training loop
tic
for n = 1:numEpochs
    [loss,gradients] = dlfeval(@myModelLoss,net,X,lblMat);
    [net,averageSqGrad] = rmspropupdate(net,gradients,averageSqGrad, eta);
    lossVec(n) = loss;
    if mod(n, 50) == 0
        fprintf('It %i, loss = %f\n', n, loss);
    end
end
toc

%% Kostenfunktion und Fehlerrate
plot(1:numEpochs, lossVec);
xlabel("Epoche"), ylabel("Kostenfunkion"), grid on; 
lblMatPred = extractdata(forward(net, X));
lblPred = onehotdecode(lblMatPred, categories(lblSpiral), 1)';
fprintf("Fehlerrate: %.1f %%\n", 100*mean(lblPred ~= lblSpiral));
