function gr = adsimple(func, xv)
    % Differenzieren mit dem komplexen Trick fuer eine Funktion mit mehreren Variablen

    assert(iscolumn(xv));    % xv muss Spaltenvektor sein
    EPS_IM = 1e-30;
    N = length(xv);
    gr = zeros(N,1); 
    for i = 1:N
        xc = complex(xv, unitvec(N, i) * EPS_IM);
        yc = func(xc);
        gr(i) = imag(yc) / EPS_IM; 
    end
end

function v = unitvec(N, i)
   v = zeros(N,1);
   v(i) = 1; 
end
