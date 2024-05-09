function xs = xsSoftmax(xMat, lblMat, wbVec)
    %xsSoftmax Kreuzentropie der Softmax-Klassifizierung
    [N,P] = size(xMat);
    C = size(lblMat, 2);
    assert(size(lblMat, 1)==N);
    
    % Die ersten P*C Elemente sind die Gewichtsmatrix
    wMat = reshape(wbVec(1:(P*C)), P, C);
    % Die hinteren C Elemente sind der Offset-Zeilenvektor
    bVec = reshape(wbVec((C*P+1):end), 1, C);
    
    [~, pMat] = applySoftmaxKlass(xMat, wMat, bVec);
    xs = -sum(lblMat .* log(pMat), 'all');

end