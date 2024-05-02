function MSE = MSEschwingung(t, y, w)
    %Berechnung der Kostenfunktion zur Schwingungsfunktion
    
     z = exp(w(1)*t) .* sin(w(2)*t);
     MSE = 1/2*mean( (z-y).^2);
end