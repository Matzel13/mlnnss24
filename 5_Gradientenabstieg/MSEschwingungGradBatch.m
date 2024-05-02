function g = MSEschwingungGradBatch(t, y, w, mbMask)
    %Minibatch-Version der Berechnung des Gradienten der Kostenfunktion der Schwingungsfunktion
    
    % Waehle die Daten des aktuellen Batches aus:
    xBatch = t(mbMask);
    yBatch = y(mbMask);

    % ab hier die gleiche Rechnung wie in MSEschwingungGrad
    z = exp(w(1)*xBatch) .* sin(w(2)*xBatch);
    Delta = (z-yBatch);
    dzdw1 = xBatch.*z;
    dzdw2 = xBatch.*exp(w(1)*xBatch).*cos(w(2)*xBatch);
    
    g = [mean(Delta.*dzdw1); mean(Delta.*dzdw2)];
end