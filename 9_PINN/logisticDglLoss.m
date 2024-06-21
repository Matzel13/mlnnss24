function [cost, grad] = logisticDglLoss(model, xColloc, f0, lambdaAB)
    f = forward(model, xColloc);
    df = dlgradient(sum(f, 'all'), xColloc, ...
        'EnableHigherDerivatives', true);
    dgl = df - f.*(1-f);
    costDgl = mean(dgl.^2);
    costAB = (f(1)-f0)^2;
    cost = costDgl + lambdaAB * costAB;
    grad = dlgradient(cost, model.Learnables);
end

