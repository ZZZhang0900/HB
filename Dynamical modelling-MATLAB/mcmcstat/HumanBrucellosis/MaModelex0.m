clear all, close all, clc
addpath('F:\SEIW-SIC\mcmcstat');

data.ydata = [
%  Time (min)     [A] (data)
0   536
1   1005
2   1611
3   2302
4   3007
5   3630
6   4147
7   4615
8   4950
9   5319
10  5673
11  5970
12  6509
13  7046
14  7755
15  8406
16  9168
17  10125
18  10979
19  11714
20  12106
21  12518
22  13035
23  13524
24  14126
25  14642
26  15618
27  16525
28  17755
29  19011
30  20119
31  21136
32  21684
33  22120
34  22918
35  23710
36  24562
37  25497
38  26860
39  28372
40  29980
41  31548
42  32968
43  34067
44  34928
45  35728
46  36631
47  37490
48  38380
49  39141
50  40830
51  42671
52  44751
53  46732
54  48445
55  49690
56  50534
57  51424
58  52313
59  53434
    ];

%% fixed parameter
global A d k sigema delta m q b beta_s

A = 5.417e6;d=0.0873;k = 1.25;
sigema = 2;delta = 0.3;m = 1/2;q = 0.6;b=0.5;beta_s = 0.130e-8;
%%
% Initial concentrations are saved in |data| to be used in sum of
% squares function.

Ss0=63.37e6;Es0=63.37e6*0.025*2/12;Is0=0.5*Es0;W0=6e6;
Sh0=9.8e6;Ih0=747;Ch0=3000;
data.y0 = [Ss0;Es0;Is0;W0;Sh0;Ih0;Ch0;536];

%%
% Refine the first Ss0=63.37e6;Es0=63.37e6*0.025*2/12;Is0=0.5*Es0;W0=6e6;
%guess for the parameters with |fminseacrh| and
% calculate residual variance as an estimate of the model error variance.
% par00=[0.015306e-8,0.0075120e-8,0.00022336e-8]';
par00 = [1.5366e-10,7.2472e-11,3.7268e-12]';
[par0,ss0] = fminsearch(@MaModelss,par00,[],data)
mse = ss0/(length(data.ydata)-1);

%%
params = {
    {'par1', par0(1), 1.4e-10, 1.7e-10}
    {'par2', par0(2), 4e-11, 10e-11}
    {'par3', par0(3), 1e-12, 9e-12}
    };

model.ssfun = @MaModelss;
model.sigma2 = mse;

options.nsimu = 4000;
options.updatesigma = 1;

%%
[results,chain,s2chain] = mcmcrun(model,data,params,options);

%%

% Then re-run starting from the results of the previous run,
% this will take couple of minutes.
options.nsimu = 10000;
[results, chain, s2chain] = mcmcrun(model,data,params,options,results);

%%
figure(2); clf
mcmcplot(chain,[],results,'chainpanel')
figure(3); clf
mcmcplot(chain,[],results,'hist')
% figure(4)
% mcmcplot(chain,[],results,'dens')
%%
% estimated Monte Carlo error of the estimates.
chainstats(chain,results)

%% plot confidential interval
burntime=6000;
%-----------------------------取多组参数-----------------------------------
N_size=length(chain)-burntime;
[mu1,sigma1,muci1,sigmaci1]=normfit(chain(burntime:end,:),0.05);
cc1=normrnd(mu1(1),sigma1(1),N_size,1);
cc2=normrnd(mu1(2),sigma1(2),N_size,1);
cc3=normrnd(mu1(3),sigma1(3),N_size,1);

%---------------------------------------------------------------------------
parameters=zeros(N_size,3);%记录每组参数
for i=1:N_size
%     i
    parameters(i,:)=[cc1(i),cc2(i),cc3(i)];  
    [T1,X11]=ode45(@MaModelode,data.ydata(:,1),data.y0,[],parameters(i,:));%利用每组参数重新运行模型
    Results(:,i)=X11(:,8);%记录每组参数下的结果
end

%---------------------------------置信区间----------------------------------
for i=1:size(T1)
    [mu1,sigma,muci,sigmaci] = normfit(Results(i,:),0.05);
    Mu(i)=mu1;
    Sigma(i)=sigma;
end
%% 置信区间绘图
fill([T1',fliplr(T1')],[Mu-1.96*Sigma,fliplr(Mu+1.96*Sigma)],[0.8706 0.9216 0.9804],'EdgeColor','none')
%% 模型和数据绘图
D=zeros(60,1);
D(1)=474;
for u=2:60
    D(u)=data.ydata(u,2)-data.ydata(u-1,2);
end
hold on
[T1,X1]=ode45(@MaModelode,0:1:59,data.y0,[],mean(chain(burntime:end,:)));%利用估计参数重新运行模型
plot(T1,X1(:,8),'b','linewidth',2.5)
xlabel('t(Time in Days)');
ylabel('Cumulative infections');
plot(data.ydata(:,1),data.ydata(:,2),'ro')

% axes('position',[0.55,0.3,0.3,0.3]);     %关键在这句！！%position是axes的其中一种property，[0.55,0.55,0.3,0.3]就是我们要设定的值分別代表左底寬高
% % plot(T1,p*sgm*X1(:,2)+0.12*X1(:,3),'g','linewidth',1.5)
% plot(T1,beta_h*(1+b*sin(pi/6*(t-1.5)))*X1(:,5)*X1(:,3)+ beta_wh*(1+b*sin(pi/6*(t-1.5)))*X1(:,5)*X1(:,4),'g','linewidth',1.5)
% hold on 
% plot(data.ydata(:,1),D,'m','linewidth',1.5)
% ylabel('Daily new cases');


figure
plot(T1,X1(:,4),'b','linewidth',2.5),hold on
plot(data.ydata(:,1),data.ydata(:,2),'ro')
