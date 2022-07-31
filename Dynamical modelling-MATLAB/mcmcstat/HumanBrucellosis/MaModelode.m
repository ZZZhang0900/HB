function ydot = MaModelode(t,y,par)

global A d k sigema delta m q b

beta_s=par(1);
beta_ws =par(1);
beta_h=par(2);
beta_wh=par(2);

% beta_ws =par(1);
% beta_h=par(2);
% beta_wh=par(3);


ydot = [
    A- beta_s*(1+b*sin(pi/6*(t-1.5)))*y(1)*y(3) - d*y(1) - beta_ws*(1+b*sin(pi/6*(t-1.5)))*y(1)*y(4);
    beta_s*(1+b*sin(pi/6*(t-1.5)))*y(1)*y(3) +  beta_ws*(1+b*sin(pi/6*(t-1.5)))*y(1)*y(4) - d*y(2) - sigema*y(2) ;
    sigema*y(2) - d*y(3);
    k*y(3) - delta*y(4);
    -beta_h*(1+b*sin(pi/6*(t-1.5)))*y(5)*y(3) - beta_wh*(1+b*sin(pi/6*(t-1.5)))*y(5)*y(4) + q*m*y(6);
    beta_h*(1+b*sin(pi/6*(t-1.5)))*y(5)*y(3) + beta_wh*(1+b*sin(pi/6*(t-1.5)))*y(5)*y(4) - m*y(6);
    (1-q)*m*y(6);
    beta_h*(1+b*sin(pi/6*(t-1.5)))*y(5)*y(3) + beta_wh*(1+b*sin(pi/6*(t-1.5)))*y(5)*y(4);
    ];