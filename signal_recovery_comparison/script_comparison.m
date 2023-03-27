%% script_comparison
%
% Description: 
%  Script to compare the IAS/GSBL and MMV-IAS/GSBL algorithm  
%
% Author: Jan Glaubitz 
% Date: Mar 27, 2023 
%


%% Parallel pool
delete(gcp);
parpool;


%% Free parameters

close all; clear all; clc ; % clean up

space = ' ';

% Parameters of the test problem 
N = 100; % length of signal
s = 20; % sparsity
C = 16; % number of signals
T = 10; % set the number of trials
mstep = 10; % stepsize in m
eps_tol = 10^(-2); % success tolerance
noise_variance = 1*10^(-6); % variance of the iid Gaussian noise added to the measurements

% Free parameters of MMV-GSBL 
R = speye(N);
c = 1; d = 10^(-4); 
% Free parameters of MMV-IAS 
r = -1; beta_IAS = 1; vartheta = 10^(-4); 


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

Err_IAS = []; Err_MMVIAS = []; % errors for IAS and MMV-IAS
Err_GSBL = []; Err_MMVGSBL = []; % errors for GSBL and MMV-GSBL
Succ_IAS = []; Succ_MMVIAS = []; % success probabilities for IAS and MMV-IAS
Succ_GSBL = []; Succ_MMVGSBL = []; % success probabilities for GSBL and MMV-GSBL


for m = mstep: mstep : N
    
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
        
        % calculate the error with weights
        error_MMVIAS(i,:) = norm(X_MMVIAS-X,'fro')/norm(X,'fro');  
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
        
        % calculate the error with weights
        error_MMVGSBL(i,:) = norm(X_MMVGSBL-X,'fro')/norm(X,'fro');
        % count the successes
        if error_MMVGSBL(i,:) <= eps_tol
            success_4 = success_4 + 1;
        end

    end
    
    Succ_IAS = [Succ_IAS ; success_1];
    Succ_MMVIAS = [Succ_MMVIAS ; success_2];
    Succ_GSBL = [Succ_GSBL ; success_3];
    Succ_MMVGSBL = [Succ_MMVGSBL ; success_4];

    Err_IAS = [Err_IAS ; mean(error_IAS)];
    Err_MMVIAS = [Err_MMVIAS ; mean(error_MMVIAS)];
    Err_GSBL = [Err_GSBL ; mean(error_GSBL)];
    Err_MMVGSBL = [Err_MMVGSBL ; mean(error_MMVGSBL)];
    
    disp(' ');
    disp(['--------------------------------- DONE ---------------------------------']);
    disp(['m = ',num2str(m)]);
    disp(['------------------------------------------------------------------------']);
    disp(' ');
    
end

%% plots

mvalues = mstep:mstep:N;

ms = 15;
ms2 = 10;
lw = 3;


fig1 = figure(1);
semilogy(mvalues,Err_IAS,'-^','Color','b','markersize',ms,'LineWidth',lw);
hold on
semilogy(mvalues,Err_MMVIAS,'--s','Color',[0 .6 0],'markersize',ms,'LineWidth',lw);
hold off

ax = gca;
ax.YMinorTick = 'off';
ax.YMinorGrid = 'off';
ax.FontSize = 26;
ax.LineWidth = 2;
ax.YGrid = 'on';
ax.XGrid = 'on';
ax.YTick = [1e-3 1e-2 1e-1 1e-0 1e+1];
ax.XTick = [0 50 100];
xlim([0 N]);
ylim([1e-3 2]);
h = legend('IAS','MMV-IAS','location','northeast');
set(h,'Interpreter','latex','color','none','fontsize',26);
xlabel('$m$','interpreter','latex','fontsize',26);
ylabel('Error','interpreter','latex','fontsize',26);


fig2 = figure(2);
plot(mvalues,Succ_IAS/T,'-^','Color','b','markersize',ms,'LineWidth',lw);
hold on
plot(mvalues,Succ_MMVIAS/T,'--s','Color',[0 .6 0],'markersize',ms,'LineWidth',lw);
hold off

ax = gca;
ax.YMinorTick = 'off';
ax.YMinorGrid = 'off';
ax.FontSize = 26;
ax.LineWidth = 2;
ax.YGrid = 'on';
ax.XGrid = 'on';
ax.YTick = [0 0.5 1.0];
ax.XTick = [0 50 100];
xlim([0 N]);
ylim([0 1]);
h = legend('IAS','MMV-IAS','location','southeast');
set(h,'Interpreter','latex','color','none','fontsize',26);
xlabel('$m$','interpreter','latex','fontsize',26);
ylabel('Success probability','interpreter','latex','fontsize',26);


fig3 = figure(3);
semilogy(mvalues,Err_GSBL,'-^','Color','b','markersize',ms,'LineWidth',lw);
hold on
semilogy(mvalues,Err_MMVGSBL,'--s','Color',[0 .6 0],'markersize',ms,'LineWidth',lw);
hold off

ax = gca;
ax.YMinorTick = 'off';
ax.YMinorGrid = 'off';
ax.FontSize = 26;
ax.LineWidth = 2;
ax.YGrid = 'on';
ax.XGrid = 'on';
ax.YTick = [1e-3 1e-2 1e-1 1e-0 1e+1];
ax.XTick = [0 50 100];
xlim([0 N]);
ylim([1e-3 2]);
h = legend('GSBL','MMV-GSBL','location','northeast');
set(h,'Interpreter','latex','color','none','fontsize',26);
xlabel('$m$','interpreter','latex','fontsize',26);
ylabel('Error','interpreter','latex','fontsize',26);


fig4 = figure(4);
plot(mvalues,Succ_GSBL/T,'-^','Color','b','markersize',ms,'LineWidth',lw);
hold on
plot(mvalues,Succ_MMVGSBL/T,'--s','Color',[0 .6 0],'markersize',ms,'LineWidth',lw);
hold off

ax = gca;
ax.YMinorTick = 'off';
ax.YMinorGrid = 'off';
ax.FontSize = 26;
ax.LineWidth = 2;
ax.YGrid = 'on';
ax.XGrid = 'on';
ax.YTick = [0 0.5 1.0];
ax.XTick = [0 50 100];
xlim([0 N]);
ylim([0 1]);
h = legend('GSBL','MMV-GSBL','color','none','location','southeast');
set(h,'Interpreter','latex','fontsize',26);
xlabel('$m$','interpreter','latex','fontsize',26);
ylabel('Success probability','interpreter','latex','fontsize',26);