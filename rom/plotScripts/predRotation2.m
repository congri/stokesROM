%to rotate predictive plot
addpath('~/matlab/toolboxes/gif_v1.0/gif');

f = gcf;
N = 180;
T = 12;
gif(strcat('~/cluster/images/multModHub/predRotationRandBC', '.gif'),...
    'DelayTime', T/N, 'frame', f);

for n = 1:N
    for i = 1:8
        ax = f.Children(i);
        ax.View = [ax.View(1) + 360/N, ax.View(2)];
    end
    gif('DelayTime', T/N);
end