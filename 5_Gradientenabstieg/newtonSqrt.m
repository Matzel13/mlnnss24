function x = newtonSqrt(a)
    x = a;
    for i = 1:300
        x = 0.5 * (x + a/x);
    end
end