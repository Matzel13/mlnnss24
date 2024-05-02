function g = MSEgeradeGrad(x, y, wb)
    %Berechnung des Gradienten der Kostenfunktion einer Regressionsgeraden in 1D
    N = length(x); 
    w = wb(1);
    b = wb(2);
    z = w*x + b;
    Delta = z-y;
    g = [1/N*Delta'*x; mean(Delta)];
end