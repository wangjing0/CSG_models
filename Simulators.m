%% Simulation of various processes
% "Reinforcement regulates timing variability" 
% Jing Wang,  jingwang.physics(a)gmail.com,  11/2020

clc; clear; close all
Wp = 0.1;
ntrials = 2e3;

Method =  'MCMC' ;

switch Method
    case 'Gaussian' % Gaussian indepedent noise
        tp = Wp.*randn(ntrials,1);
        reward_window = Wp;
        reward = (abs(tp)<reward_window ).* (1-abs(tp)/reward_window);
    case 'GP' % Gaussian process
        alpha = 1;
        sig=1;
        ssig0 =.5;
        l1= 20; % GP, slow term
        l2= 2; % reward, fast term
        [tp,~,~,reward] = RSGPsimulator(ntrials,Wp,sig,sig0,l1,l2,alpha);
    case 'RSGP'% Reward Sensitive Gaussian Process 
        clear x;
        alpha = .5; % ratio of var( gaussian process) / total variance
        sig=1; % total variance
        sig0 = .5; % variance of private noise
        l1= 20; % GP, slow term
        l2= 2; % reward, fast term
        [tp,~,~,reward] = RSGPsimulator(ntrials,Wp,sig,sig0,l1,l2,alpha);
    case 'DS'% Directed search
        alpha = .5; %default parameters; gain of directional adjustment
        Wp   = .1;
        We   = .01;
        STOP_AT_RUNAWAY = false;
        [tp, te, Reward]= DSsimulator(ntrials,STOP_AT_RUNAWAY,alpha,Wp,We);
        
    case 'MCMC'% Markov chain Monte Carlo
        beta = 100 ; % MCMC: large - low temperature, small - high temperature
        Wp   = .1;
        We   = .01;
        STOP_AT_RUNAWAY = false;
        [tp,te,Reward]= MCMCsimulator(ntrials,STOP_AT_RUNAWAY,beta,Wp,We);       
end