function [loss,gradients] = myModelLoss(net,X,T)
   Y = forward(net,X);
   loss =  crossentropy(Y,T);
   gradients = dlgradient(loss, net.Learnables);
end