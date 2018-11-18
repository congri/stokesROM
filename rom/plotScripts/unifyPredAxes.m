f = gcf;
z1 = [-4e5 -3e6 -5e5 -3e5 -1e5 -1e5];
z2 = [6e5 3e6 4.5e5 4.5e5 1e5 1.5e5];
for i = 1:6
    ax = f.Children(14 - 2*i);
    ax.ZLim(1) = z1(i);
    ax.ZLim(2) = z2(i);
end