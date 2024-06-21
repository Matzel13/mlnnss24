function [cost, grads] = ballMotionLoss(model, tDaten, sDaten, tColoc, mu)
    
    g = 9.81;
    fDaten = forward(model, tDaten);
    assert(isequal(size(fDaten), size(sDaten)));
    costData = mean( (fDaten - sDaten).^2, 'all');
  
    Nc = length(tColoc);
    fColoc = forward(model, tColoc);
    sx = fColoc(1,:);
    sy = fColoc(2,:);
   
    vx = dlgradient(sum(sx, 'all'), tColoc, 'EnableHigherDerivatives', true);
    ax = dlgradient(sum(vx, 'all'), tColoc, 'EnableHigherDerivatives', true);
    vy = dlgradient(sum(sy, 'all'), tColoc, 'EnableHigherDerivatives', true);
    ay = dlgradient(sum(vy, 'all'), tColoc, 'EnableHigherDerivatives', true);
    vabs = sqrt(vx.^2 + vy.^2);

    dglx = ax + mu*vabs.*vx;
    costDglx = 1/Nc * sum(dglx.*dglx);
    dgly = ay + mu*vabs.*vy + g;
    costDgly = 1/Nc * sum(dgly.*dgly);
    costAB = 10 * (sx(1) - sDaten(1,1))^2 + (sy(1) - sDaten(2,1))^2;
    lambda = 1; 
    cost = costData + costAB + lambda * (costDglx + costDgly); 
    grads = dlgradient(cost, {model.Learnables, mu});
    grad = grads{1};
    mask = model.Learnables.Parameter == "Weights" | model.Learnables.Parameter == "Bias";
    lambdaWeights = 0.001;
    grad(mask,:) = dlupdate(@(g,w) g + lambdaWeights*w, grad(mask,:), model.Learnables(mask,:));
    grads{1} = grad; 
   
end

