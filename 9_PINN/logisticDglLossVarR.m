function [cost, grads] = logisticDglLossVarR(model, xColloc, ...
        xData, yData, R, lambdaDat, lambdaAB)
    
    % N = length(xColloc);
    fColloc = forward(model, xColloc);
    fData = forward(model, xData);
    dfColloc = dlgradient(sum(fColloc, 'all'), xColloc, 'EnableHigherDerivatives', true);
    dgl = dfColloc - R * fColloc.*(1-fColloc);
    costDgl = mean(dgl.^2);
    costAB = (fData(1)-yData(1))^2;
    costData = mean( (fData - yData).^2);
    cost = costDgl + lambdaDat*costData + lambdaAB * costAB ;
    grads = dlgradient(cost, {model.Learnables, R});

end