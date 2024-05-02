function [cost, grad] = dlMSE(mdl, x, y)
   % MSE-Kostenfunktion fuer dl-Toolbox
   ypred = forward(mdl, x);
   cost =  1/2 * mean( (ypred - y).^2);
   grad = dlgradient(cost, mdl.Learnables);
end
