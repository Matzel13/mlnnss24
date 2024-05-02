function MSE = MSEgerade(x, y, wb)
    %Berechnung der Kostenfunktion einer Regressionsgeraden in 1D
    
    N = length(x);
    w = wb(1);
    b = wb(2);
    z = w*x + b;
    MSE = 1/(2*N) * sum( (z-y).^2);
end