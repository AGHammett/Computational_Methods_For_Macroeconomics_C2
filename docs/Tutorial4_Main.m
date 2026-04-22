% Tutorial4_Main.m
% Replicate Prescott's paper, Main Program
% This is a modified version of the program in Greenwood+Marto, Ch4
clear; clc; close all;

%% Step 1: load data
data.all       = readtable('data_prescott.csv');  % load data

data.country   = data.all.country(1:7);  % Country names
data.c2y9396   = data.all.c2y(1:7);      % Cons/output ratio
data.tau9396   = data.all.tau(1:7);      % Effective tax rate
data.hdata9396 = data.all.h(1:7);        % Hours data 
data.c2y7074   = data.all.c2y(8:14);     % Cons/output ratio
data.tau7074   = data.all.tau(8:14);     % Effective tax rate
data.hdata7074 = data.all.h(8:14);       % Hours data 

%% Step 2: Set predeteremined parameter(s)
theta = 0.32;      % production parameter, capital share

%% Step 3: Calibrate the preference parameter alpha
alfa = fminsearch (@(a) LossFcn(a,theta,data) , 1.54 ) ; % Call up minimization routine (Nelder-Mead here)
fprintf('The calibrated alpha is %2.3f \n\n',alfa); 

%% Step 4: Plot the results
% Calculate model hours 
hmodel9396 = 100.*(1-theta ) ./ ( alfa./(1-data.tau9396) .* data.c2y9396  + (1-theta ) ) ;
hmodel7074 = 100.*(1-theta ) ./ ( alfa./(1-data.tau7074) .* data.c2y7074  + (1-theta ) ) ;

% Plot the data versus model for the 7 countries
makefigures ( 1, data.hdata9396 , hmodel9396 , data.country , 'Fit of Model: Hours, 1993-1996' )
makefigures ( 2, data.hdata7074 , hmodel7074 , data.country , 'Fit of Model: Hours, 1970-1974' )

%% Counterfactual Analysis
% Remember G7 Countres: GER, FRA, ITA, CAN, GBR, JPN, USA
ID           = 5;     % Select your base country
counter_ID   = 6;     % Select the country to use counterfactual analysis

% Counterfactual Analysis
h_base    = hmodel9396(ID);   % Hours worked of the base country predicted by the model
h_counter = 100.*(1-theta ) ./ ( alfa./(1-data.tau9396(counter_ID))...
                            .* data.c2y9396(ID)  + (1-theta ) ) ; % Hours prediction if the country with ID changed its tax rate to the one of the country with counter_ID
fprintf('Hours worked for %1s predicted by the model: %2.3f \n',data.country{ID},h_base); 
fprintf('Counterfactual hours for %1s using the tax system of %1s: %2.3f \n\n',data.country{ID},data.country{counter_ID},h_counter); 

