function [Tp,Te,Reward]=MCMCsimulator(ntrials,STOP_AT_RUNAWAY, varargin)
if numel(varargin) > 0
    beta = varargin{1};
    Wp = varargin{2};
    We = varargin{3};
else %default parameters
    beta = 100; % inverse temperature: 1/kT
    Wp   = .1;
    We   = .05;
end

k_reward = Wp+We; dk_reward = 0.01;
k_reward_min=0.01; k_reward_max=.5;
Nmax  = ntrials;
Te = nan(Nmax,1);
Tp = nan(Nmax,1);
Vp = nan(Nmax,1); % posterior value, value = - cost
Vc = nan(Nmax,1); % current highest value
Reward = nan(Nmax,1);
rng(0);

Te(1) = 0;      k_reward = Wp+We;
exe_noise = Wp*randn(Nmax,1);
centr_noise = We*randn(Nmax,1);
P = rand(Nmax,1);
for i=1:Nmax
    Te_try = Te(i)+ centr_noise(i);
    Tp(i) = (Te_try +1).*(1+ exe_noise(i));%
    Vc(i) = (abs(Te(i))<k_reward ).* (1 - abs(Te(i))./(k_reward));
    Vp(i) = (abs(Tp(i)-1)< k_reward).* (1 - abs(Tp(i)-1)./(k_reward));
    Reward(i) = Vp(i);
    p_accept = exp(beta.*Vp(i))./(exp(beta.*Vp(i)) + exp(beta.*Vc(i)) );% prob of accepting the reward as positive, and change the best estimate
    if STOP_AT_RUNAWAY && (k_reward>= k_reward_max)
        break;
    end % stop at runaway
    if i<Nmax
        if  Reward(i)>0
            k_reward=max(k_reward_min, k_reward-dk_reward);
        else
            k_reward=min(k_reward_max, k_reward+dk_reward);
        end
        if (p_accept > P(i))
            Te(i+1) = Te_try ;%Te+
        else
            Te(i+1) = Te(i); % Te*
        end
    end
end
