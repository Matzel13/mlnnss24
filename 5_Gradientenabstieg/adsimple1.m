function gr = adsimple1(func, x)
    % Differenzieren mit dem komplexen Trick fuer eine Funktion mit einer Variablen
    % Toy-Version von AD per Operator-Ueberladung
    
    EPS_IM = 1e-30;
    xc = complex(x, EPS_IM);
    yc = func(xc);
    gr = imag(yc) / EPS_IM;
end
