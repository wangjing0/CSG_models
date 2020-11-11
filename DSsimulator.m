function [Tp, Te, Reward]= DSsimulator(ntrials,STOP_AT_RUNAWAY,varargin)
if numel(varargin) > 0
    alpha = varargin{1};
    Wp = varargin{2};
    We = varargin{3};
else %default parameters
    alpha = .5;
    Wp   = .11;
    We   = .01;
end

k_reward_min=0.05;
k_reward_max = 0.3;
dk_reward = 0.01;

Nmax = ntrials;
Te = nan(Nmax,1); % estimation
Tp = nan(Nmax,1); % production
Reward = nan(Nmax,1);
rng(0);

Te(1) = 0;  k_reward = Wp+We;
exe_noise = Wp*randn(Nmax,1);
est_noise = We*randn(Nmax,1);
est_shift = 0;
for i=1:Nmax
    Tp(i) = (1+Te(i)).*(1 + exe_noise(i)); % scalar noise ,Tp = Tp/Ts
    Reward(i) = (abs( Tp(i)-1)<k_reward ).* (1 - abs( Tp(i)-1)./(k_reward));
    if STOP_AT_RUNAWAY && (k_reward>= k_reward_max)
        break;
    end % stop at runaway
    
    if Reward(i)>0
        k_reward=max(k_reward_min, k_reward-dk_reward);
        
    else
        k_reward=min(k_reward_max, k_reward+dk_reward);
    end
    
    if i<Nmax
        if i==1
            est_shift =  0;
        else
            d_te = Te(i) - Te(i-1) ;
            d_rew = Reward(i) - Reward(i-1);
            est_shift = d_te.* d_rew;
        end
        Te(i+1) = Te(i) + alpha.* est_shift + est_noise(i);
    end
end

