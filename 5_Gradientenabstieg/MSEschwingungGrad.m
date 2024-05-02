function g = MSEschwingungGrad(t, y, w)
    %Berechnung des Gradienten der Kostenfunktion der Schwingungsfunktion
    
    z = exp(w(1)*t) .* sin(w(2)*t);
    Delta = (z-y);
    dzdw1 = t.*z;
    dzdw2 = t.*exp(w(1)*t).*cos(w(2)*t);
    
    g = [mean(Delta .* dzdw1); mean(Delta.*dzdw2)];
end