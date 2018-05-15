%This is a test script calling python from within matlab

addpath('./pythonEmulator')

% execute python as shell command
out = evalc('system(''/home/constantin/anaconda3/envs/dfa/bin/python /home/constantin/python/projects/stokesEquation/rom/pythonEmulator/pyPoisson.py'')')


% %% Reload the python modules
% clear classes;
% mod_test = py.importlib.import_module('pythonEmulator.pyInMatlabTest');
% % mod_pyEm = py.importlib.import_module('pythonEmulator.pyPoisson');
% py.importlib.reload(mod_test);
% % py.importlib.reload(mod_pyEm);
% 
% str = 'printing this string';
% out = py.pythonEmulator.pyInMatlabTest.testFun(str)
% 
% cond = 1 + lognrnd(0, 1, 32, 1);
% % out2 = py.pythonEmulator.pyPoisson.forwardSolver(cond)