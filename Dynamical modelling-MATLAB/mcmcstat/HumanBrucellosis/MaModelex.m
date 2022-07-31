clear all, close all, clc
addpath('F:\SEIW-SIC\mcmcstat');

data_daily.ydata = [   
%  Time (min)     [A] (mole/liter)
1   536
2   469
3   606
4   691
5   705
6   623
7   517
8   468
9   335
10   369
11  354
12  297
13  539
14  537
15  709
16  651
17  762
18  957
19  854
20  735
21  392
22  412
23  517
24  489
25  602
26  516
27  976
28  907
29  1230
30  1256
31  1108
32  1017
33  548
34  436
35  798
36  792
37  852
38  935
39  1363
40  1512
41  1608
42  1568
43  1420
44  1099
45  861
46  800
47  903
48  859
49  890
50  761
51  1689
52  1841
53  2080
54  1981
55  1713
56  1245
57  844
58  890
59  889
60  1121];

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
59  53434];

%% fixed parameter
global A d k sigema delta m q b

A = 5.417e6;d=0.087;k = 1.25; b=0.5;
sigema = 2;delta = 0.3;m = 1/2;q = 0.6;
%%
% Initial concentrations are saved in |data| to be used in sum of
% squares function.

Ss0=63.37e6;Es0=63.37e6*0.025*2/12;Is0=0.5*Es0;W0=6e6;
Sh0=9.8e6;Ih0=747;Ch0=5000;
data.y0 = [Ss0;Es0;Is0;W0;Sh0;Ih0;Ch0;536];

%%
% Refine the first guess for the parameters with |fminseacrh| and
% calculate residual variance as an estimate of the model error variance.
% par00 =[3.982e-9,1.21e-13]';
par00 =[0.3786e-09,0.0281e-9]';
% par00 = [8.22114e-10,3.93347e-10,5.662e-12,6.78e-13,0.411252788557707]';
[par0,ss0] = fminsearch(@MaModelss,par00,[],data)
mse = ss0/(length(data.ydata)-1);

%%
params = {
    {'par1', par0(1), 1e-10, 10e-10}
    {'par2', par0(2), 0.8e-11, 9e-11}
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
%---------------------------------------------------------------------------
parameters=zeros(N_size,2);%记录每组参数
for i=1:N_size
%     i
    parameters(i,:)=[cc1(i),cc2(i)];  
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




[T1,X1]=ode45(@MaModelode,1:1:60,data.y0,[],mean(chain(burntime:end,:)));%利用估计参数重新运行模型
z=zeros(1,60);
z(1)=X1(1,8);
for kk=2:1:60
z(kk)=X1(kk,8)-X1(kk-1,8);

end

figure
t=1:1:60;
plot(t,z,'b','linewidth',2)
xlabel('Time (month)');
ylabel('Number of newly infected population');
hold on
plot(data_daily.ydata(:,1),data_daily.ydata(:,2),'ro')
ylim([0 2500])
xlim([-2 62])




figure
plot(T1,X1(:,8),'b','linewidth',2.5),hold on
plot(data.ydata(:,1),data.ydata(:,2),'ro')
