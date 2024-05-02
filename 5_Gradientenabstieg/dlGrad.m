function g = dlGrad(func, x)
    % Wrapper im Matlabs AD-Funktionalitaet zur Berechnung des 
    % Gradienten einer Funktion

    gdl = dlfeval(@(x) evalGrad(func, x), dlarray(x));
    g = extractdata(gdl);
end

function g = evalGrad(func, dlx)
    dly = func(dlx);
    g = dlgradient(dly, dlx);
end