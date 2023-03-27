%% script_phaseTransition
%
% Description: 
%  Script to compute empirical success probabilities and plot the corresponding 
%  phase transition diagrams 
%
% Author: Jan Glaubitz 
% Date: Jan 22, 2023 
%

%% Parallel pool
delete(gcp);
parpool;


%% Free parameters

close all; clear all; clc ; % clean up

space = ' ';

% Parameters of the test problem 
N = 200; % length of signal
C = 4; % number of signals
T = 10; % set the number of trials
stepsize = 4; % stepsize in m and s
eps_tol = 10^(-2); % success tolerance
noise_variance = 1*10^(-6); % variance of the iid Gaussian noise added to the measurements

R = speye(N);
% Free parameters of MMV-GSBL 
c = 1; d = 10^(-4); 
% Free parameters of MMV-IAS 
r = -1; beta_IAS = -1; vartheta = 10^(-4); 


%% Initialize arrays

X = zeros(N,C); % signal matrix
X_IAS = zeros(N,C); % recovered signals for IAS
X_MMVIAS = zeros(N,C); % recovered signals for MMV-IAS
X_GSBL = zeros(N,C); % recovered signals for GSBL
X_MMVGSBL = zeros(N,C); % recovered signals for MMV-GSBL

error_IAS = zeros(T,1); % error for each trial for IAS
error_MMVIAS = zeros(T,1); % error for each trial for MMV-IAS
error_GSBL = zeros(T,1); % error for each trial for GSBL
error_MMVGSBL = zeros(T,1); % error for each trial for MMV-GSBL

%s_values = stepsize:stepsize:N; % vector containing the values for s
m_values = stepsize:stepsize:N; % vector containing the values for m

%Succ_IAS = zeros(length(m_values),length(s_values)); % success probabilities for IAS
%Succ_MMVIAS = zeros(length(m_values),length(s_values)); % success probabilities for MMV-IAS
%Succ_GSBL = zeros(length(m_values),length(s_values)); % success probabilities for GSBL 
%Succ_MMVGSBL = zeros(length(m_values),length(s_values)); % success probabilities for MMV-GSBL

% read matrices from files
Succ_IAS = readmatrix('Succ_IAS.txt'); % success probabilities for IAS
Succ_MMVIAS = readmatrix('Succ_MMVIAS.txt'); % success probabilities for MMV-IAS
Succ_GSBL = readmatrix('Succ_GSBL.txt'); % success probabilities for GSBL 
Succ_MMVGSBL = readmatrix('Succ_MMVGSBL.txt'); % success probabilities for MMV-GSBL

s_values = 180:stepsize:N; % vector containing the values for s

for s = s_values 
for m = m_values
    
    % initialize various arrays/parameters
    success_1 = 0; success_2 = 0; 
    success_3 = 0; success_4 = 0; 
    
    % loop over the number of trials
    for i = 1 : T
        
        Y = []; % measurements Y 
        
        % random choose A as m*N subsampled DCT
        idx = randperm(N);
        idx = idx(1:m);
        A = dctmtx(N); 
        A = A(idx,:); 
        
        % randomly generate the support set b
        b = randperm(N,s)';
        
        % Generate signals and measurements
        for j = 1:C
            
            % Generate signal
            a = randn(s,1);
            x_aux = sparse(b , 1 , a, N,1);
            x{j} = full(x_aux);
            X(:,j) = x{j}; 

            % Calculate measurements 
            F{j} = A; 
            noise_variances{j} = noise_variance; 
            noise{j} = sqrt(noise_variance)*randn(size(F{j},1),1); % iid normal noise
            y{j} = F{j}*x{j} + noise{j}; 
            
        end
        
        
        % recover each signal separately using IAS 
        parfor j = 1:C            
            [x_IAS{j}, theta, history] = IAS_1d( F{j}, y{j}, noise_variances{j}, R, r, beta_IAS, vartheta, 1 );
            X_IAS(:,j) = x_IAS{j};            
        end      
        error_IAS(i,:) = norm(X-X_IAS,'fro')/norm(X,'fro'); % calculate the error
        
        % counts the number of success
        if error_IAS(i,:) <= eps_tol
            success_1 = success_1 + 1;
        end
    
        
        % Use MMV-IAS 
        [x_MMVIAS, theta, history] = MMV_IAS_1d( C, F, y, noise_variances, R, r, beta_IAS, vartheta, 1 );
        parfor j = 1:C         
            X_MMVIAS(:,j) = x_MMVIAS{j};            
        end
        error_MMVIAS(i,:) = norm(X_MMVIAS-X,'fro')/norm(X,'fro'); % calculate the error
        
        % count the successes
        if error_MMVIAS(i,:) <= eps_tol
            success_2 = success_2 + 1;
        end                    
        

        % recover each signal separately using GSBL 
        parfor j = 1:C            
            [x_GSBL{j}, theta, history] = GSBL_1d( F{j}, y{j}, noise_variances{j}, R, c, d, 1 );
            X_GSBL(:,j) = x_GSBL{j};          
        end           
        error_GSBL(i,:) = norm(X-X_GSBL,'fro')/norm(X,'fro'); % calculate the error
        
        % counts the number of success
        if error_GSBL(i,:) <= eps_tol
            success_3 = success_3 + 1;
        end
    
        
        % Use MMV-GSBL  
        [x_MMVGSBL, theta, history] = MMV_GSBL_1d( C, F, y, noise_variances, R, c, d, 1 );
        parfor j = 1:C           
            X_MMVGSBL(:,j) = x_MMVGSBL{j};            
        end
        error_MMVGSBL(i,:) = norm(X_MMVGSBL-X,'fro')/norm(X,'fro'); % calculate the error
        
        % count the successes
        if error_MMVGSBL(i,:) <= eps_tol
            success_4 = success_4 + 1;
        end

    end
    
    % Compute empirical success probabilities
    Succ_IAS(s/stepsize,m/stepsize) = success_1/T;
    writematrix(Succ_IAS,'Succ_IAS.txt','Delimiter','tab');
    Succ_MMVIAS(s/stepsize,m/stepsize) = success_2/T;
    writematrix(Succ_MMVIAS,'Succ_MMVIAS.txt','Delimiter','tab');
    Succ_GSBL(s/stepsize,m/stepsize) = success_3/T;
    writematrix(Succ_GSBL,'Succ_GSBL.txt','Delimiter','tab');
    Succ_MMVGSBL(s/stepsize,m/stepsize) = success_4/T;
    writematrix(Succ_MMVGSBL,'Succ_MMVGSBL.txt','Delimiter','tab');

    % Show progress 
    disp(['s = ',num2str(s),space,'m = ',num2str(m)]);
    
end
end
 

%% Plot phase transitions

fig1 = figure(1);
pcolor(Succ_IAS);
colorbar
ax = gca;
axticks =  1:N/stepsize;
axticklabels = axticks*stepsize; 
xlabel('$m/N$','FontSize', 26,'Interpreter','LaTex')
ylabel('$s/N$','FontSize', 26,'Interpreter','LaTex')
%titlename = ['IAS'];
%title(titlename,'FontSize', 18,'Interpreter','LaTex');

fig2 = figure(2);
pcolor(Succ_MMVIAS);
colorbar
ax = gca;
axticks =  1:N/stepsize;
axticklabels = axticks*stepsize; 
xlabel('$m/N$','FontSize', 26,'Interpreter','LaTex')
ylabel('$s/N$','FontSize', 26,'Interpreter','LaTex')
%titlename = ['IAS'];
%title(titlename,'FontSize', 18,'Interpreter','LaTex');

fig3 = figure(3);
pcolor(Succ_GSBL);
colorbar
ax = gca;
axticks =  1:N/stepsize;
axticklabels = axticks*stepsize; 
xlabel('$m/N$','FontSize', 26,'Interpreter','LaTex')
ylabel('$s/N$','FontSize', 26,'Interpreter','LaTex')
%titlename = ['IAS'];
%title(titlename,'FontSize', 18,'Interpreter','LaTex');

fig4 = figure(4);
pcolor(Succ_MMVGSBL);
colorbar
ax = gca;
axticks =  1:N/stepsize;
axticklabels = axticks*stepsize; 
xlabel('$m/N$','FontSize', 26,'Interpreter','LaTex')
ylabel('$s/N$','FontSize', 26,'Interpreter','LaTex')
%titlename = ['IAS'];
%title(titlename,'FontSize', 18,'Interpreter','LaTex');

