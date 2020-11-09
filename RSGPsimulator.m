function [Tp,mTe,sigTe,Reward]= RSGPsimulator(Nmax,Wp,sig,sig0,l1,l2,alpha)

%Wp      = .1;
sig_g   = sig*Wp;
%l1      = 20; % correlation length, in unit of trials
%l2      = 2; % correlation length, in unit of trials
%Alpha   = 1;%[1, 0.5, 0.0];
sig_g0  = sig0*Wp;
dk_reward = 0.01;

k_reward = Wp;
k_reward_min=0.05; k_reward_max=0.3;
kernelfunction ='squaredexponential';

% GP kernel
[covInd1,covInd2] = meshgrid((1:Nmax),(1:Nmax));

switch kernelfunction
    case 'squaredexponential'
        cov_g1 = sig_g^2 * exp(- (covInd1-covInd2).^2 ./(2.*l1.^2)) ;
        cov_g2 = sig_g^2 * exp(- (covInd1-covInd2).^2 ./(2.*l2.^2)) ;
        cov_g3 = sig_g0^2.* eye(Nmax);
    case 'exp'
        cov_g = sig_g^2 * exp(- (abs(covInd1-covInd2)) ./l1) + sig_g0^2.* eye(Nmax);
    case  'equal'
        cov_g = sig_g^2 * ones(size(covInd1)) + sig_g0^2.* eye(Nmax);
    case 'whitenoise'
        cov_g = sig_g^2.* eye(Nmax);
    case 'besselk'
        nu=l-1/2; % order of, modified Bessel function of second kind
        xi_xj = sqrt(2.*nu).*abs(covInd1-covInd2)./l1;
        cov_g = sig_g^2.*(1./gamma(nu)./2.^(nu-1)).*((xi_xj).^nu).*besselk(nu,xi_xj);
        cov_g(logical(eye(Nmax)))=sig_g^2 + sig_g0^2;
end

acLength=ceil(2.5*max(l1,l2)); % max length of correlation to be considered

Te=nan(Nmax,1);
Tp=nan(Nmax,1);
mTe=nan(Nmax,1); mTe(1)= 0;
sigTe=nan(Nmax,1); sigTe(1)=sig_g^2 + sig_g0^2;
Reward=nan(Nmax,1);
rng(1); % reset the random number generator

rand1=randn(Nmax,1);

% updating trial by trial
for i=1:Nmax
    Te(i) = mTe(i) +  rand1(i).*sqrt(sigTe(i) ) ;
    Tp(i) = Te(i) + 1;
    Reward(i) = (abs(Te(i))< k_reward ).* (1 - abs(Te(i))./(k_reward));
    
    if dk_reward>0 % reward window on staircase
        if Reward(i)>0
            k_reward=max(k_reward_min, k_reward-dk_reward);
        else
            k_reward=min(k_reward_max, k_reward+dk_reward);
        end
    end
    
    if i < Nmax
        clear K* Te_
        K1  = cov_g1(max(1,i-acLength):i,max(1,i-acLength):i); % ixi cov of known
        K_1 = cov_g1(i+1,max(1,i-acLength):i); % 1xi
        K__1= cov_g1(i+1,i+1);  % 1x1 cov of predict
        K2  = cov_g2(max(1,i-acLength):i,max(1,i-acLength):i); % ixi cov of known
        K_2 = cov_g2(i+1,max(1,i-acLength):i); % 1xi
        K__2= cov_g2(i+1,i+1);  % 1x1 cov of predict
        K3  = cov_g3(max(1,i-acLength):i,max(1,i-acLength):i);
        K__3= cov_g3(i+1,i+1);
        Te_ = Te(max(1,i-acLength):i);
        
        indx= find(Reward(max(1,i-acLength):i)==0);% only use rewarded trials for update
        for k = 1:length(indx)
            kd = min(indx(k)+1,size(K2,1));
            K2(indx(k),kd:end) = 0;  K2(kd:end,indx(k)) = 0;
        end
        K2(logical(eye(size(K2))))=cov_g2(i,i) ;
        K_2(indx) = 0;
        
        % mu and sigma
        sigTe(i+1)= (alpha.*K__1+ (1-alpha).*K__2 + K__3) - ...
            (alpha*K_1+ (1-alpha).*K_2)*inv(alpha*K1 + (1-alpha).*K2 + K3)*(alpha*K_1 + (1-alpha).*K_2)' ;
        
        mTe(i+1)  =(alpha.*K_1 + (1-alpha).*K_2) * inv(alpha*K1 + (1-alpha).*K2 + K3) * Te_;
    end
end
