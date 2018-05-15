function [mx, f_mx] = max_fun(fun, x0)
%Maximize function

    function [f_neg, d_f_neg] = neg_fun(f, x)
        [f_temp, d_f_temp] = f(x);
        f_neg = - f_temp;
        d_f_neg = - d_f_temp;
    end

f_opt = @(xx) neg_fun(fun, xx);

opts = optimoptions(@fminunc,'Display','off', ...
    'MaxFunctionEvaluations', 1e4, 'Algorithm', 'trust-region', ...
    'SpecifyObjectiveGradient', true, 'FunctionTolerance', 1e-18);

[mx, f_mx] = fminunc(f_opt, x0, opts);

end

