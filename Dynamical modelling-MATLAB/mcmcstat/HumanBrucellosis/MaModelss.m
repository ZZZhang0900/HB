function ss = MaModelss(par,data)
% sum-of-squares for Himmelblau 9.9

time = data.ydata(:,1);
Aobs = data.ydata(:,2);
y0   = data.y0;

[t,y] = ode45(@MaModelode,time,y0,[],par);
Amodel = y(:,8);

ss = sum((Aobs-Amodel).^2);
