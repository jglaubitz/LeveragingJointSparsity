%% script_pMRI_errors
%
% Description: 
%  Script to perform an error analysis for multi-coil parallel magnetic resonance imaging (mPRI)
%
% Author: Jan Glaubitz 
% Date: Mar 31, 2023 
%

clear; close all; clc; % clean up
%warning('off','all') % in case any of the warnings become too anoying 


%% Free parameters 

% Free parameters of the problem 
N = 256; % number of pixels in each direction 
C = 4; % number of coils 
lines = 10:10:100; % number of angles
sigma_sq = 10^(-3); % noise variance
rng('default'); rng(1,'twister'); % to make the results reproducable

% Free parameters of the methods 
reg_type = 'TV'; % type of regualrization (TV = total variation, PA = polynomial annihilation) 
order = 1;
c_GSBL = 1; d_GSBL = 10^(-3); % Free parameters of GSBL 
r = -1; beta_IAS = 1; vartheta = 10^(-3); % Free parameters of IAS 


%% Construct coils and images 

% Set up the brain image 
%FOV = 0.256; % Field-of-view (FOV) width
%mxsize = N*[1 1];
%pixelsize = FOV./mxsize;
%DefineBrain;
%Brain.FOV = FOV*[1, 1];

% Discretize the brain 
%X = RasterizePhantom(Brain,mxsize); % reference brain image 
X = phantom('Modified Shepp-Logan',N);
x = reshape(X,N*N,1); 
figure(1)
imshow(X);
colormap(gray)
%colormap(hot)

% Recovered coil images
xCrecLS = zeros(N*N,C); % recovered vectorized coil images using LS
xCrecIAS = zeros(N*N,C); % recovered vectorized coil images using IAS
xCrecMMVIAS = zeros(N*N,C); % recovered vectorized coil images using MMV-IAS
xCrecGSBL = zeros(N*N,C);% recovered vectorized coil images using GSBL
xCrecMMVGSBL = zeros(N*N,C); % recovered vectorized coil images using MMV-GSBL

% Errors and SER of recovered coil images
error_coil_LS = []; % LS
error_coil_IAS = []; % IAS
error_coil_MMVIAS = []; % MMV-IAS
error_coil_GSBL = []; % GSBL 
error_coil_MMVGSBL = []; % MMV-GSBL

% Errors of recovered overall images
error_overall_LS = []; % LS
error_overall_IAS = []; % IAS
error_overall_MMVIAS = []; % MMV-IAS
error_overall_GSBL = []; % GSBL 
error_overall_MMVGSBL = []; % MMV-GSBL


% for loop over number of angles/lines
for l = lines

    l
    % Construct vectorized coils, coil images, and measurements 
    for c=1:C 
        % Construct sampling operator  
        [idx1{c},idx2{c},R{c}] = MMV_SampMatrixRadial(N,l,c,C);
        m{c} = nnz(R{c});
        opF{c} = @(x,mode) partialFourierShift2D(idx1{c},idx2{c},N,x,mode); % forward operator
        noise{c} = randn(m{c},1); % iid zero-mean normal noise 
        noise_var{c} = sigma_sq/(norm(noise{c})^2); % we use the same noise variance for all MMVs 
        noise{c} = sqrt(noise_var{c})*noise{c}; % iid zero-mean normal noise 
        y{c} = opF{c}(x,'notransp') + noise{c}; % compute measurements 
    end
    % Construct function handle for the regularization operator
    opR = @(x,flag) reg_op( N, x, reg_type, order, flag ); % regularization operator
    
    
    %% Separate reconstructions using LS, IAS, and GSBL 
    
    parfor c=1:C  
        xCrecLS(:,c) = lsqr( opF{c}, y{c} ); % least squares approximation
        [xCrecIAS(:,c), theta, history] = IAS( opF{c}, y{c}, noise_var{c}, opR, r, beta_IAS, vartheta, 1 ); % Use IAS
        [xCrecGSBL(:,c), theta, history] = GSBL( opF{c}, y{c}, noise_var{c}, opR, c_GSBL, d_GSBL, 1 ); % Use GSBL
    end 
    
    
    %% Joint reconstruction using MMV-IAS and -GSBL 
    [x_MMVIAS, theta, history] = MMV_IAS( C, opF, y, noise_var, opR, r, beta_IAS, vartheta, 1 );
    [x_MMVGSBL, theta, history] = MMV_GSBL( C, opF, y, noise_var, opR, c_GSBL, d_GSBL, 1 );
    % transform into matrix for subsequent comparision and plotting routines
    parfor c=1:C  
        xCrecMMVIAS(:,c) = x_MMVIAS{c}; % MMV-IAS
        xCrecMMVGSBL(:,c) = x_MMVGSBL{c}; % MMV-GSBL
    end 
    
    
    %% Compute the overall image from the seperate images  
    x_LS = sum(xCrecLS')'/C; % average of separate images
    x_IAS = sum(xCrecIAS')'/C; % average of separate images
    x_MMVIAS = sum(xCrecMMVIAS')'/C; % average of separate images
    x_GSBL = sum(xCrecGSBL')'/C; % average of separate images
    x_MMVGSBL = sum(xCrecMMVGSBL')'/C; % average of separate images
    
    
    %% Compute errors and SER
    
    % Coil error and SER for LS 
    error = norm(xCrecLS-x,'fro')/norm(x,'fro'); % relative error 
    error_coil_LS = [error_coil_LS, error]; % store relative error 
    
    % Coil error and SER for IAS 
    error = norm(xCrecIAS-x,'fro')/norm(x,'fro'); % relative error 
    error_coil_IAS = [error_coil_IAS, error]; % store relative error 
    
    % Coil error and SER for MMV-IAS 
    error = norm(xCrecMMVIAS-x,'fro')/norm(x,'fro'); % relative error 
    error_coil_MMVIAS = [error_coil_MMVIAS, error]; % store relative error 
    
    % Coil error and SER for GSBL 
    error = norm(xCrecGSBL-x,'fro')/norm(x,'fro'); % relative error 
    error_coil_GSBL = [error_coil_GSBL, error]; % store relative error  
    
    % Coil error and SER for MMV-GSBL 
    error = norm(xCrecMMVGSBL-x,'fro')/norm(x,'fro'); % relative error 
    error_coil_MMVGSBL = [error_coil_MMVGSBL, error]; % store relative error 
    
    % overall error and SER for LS 
    error = norm(x_LS-x)/norm(x); % relative error 
    error_overall_LS = [error_overall_LS, error]; % store relative error  
    
    % overall error and SER for IAS 
    error = norm(x_IAS-x)/norm(x); % relative error 
    error_overall_IAS = [error_overall_IAS, error]; % store relative error 
    
    % overall error and SER for MMV-IAS 
    error = norm(x_MMVIAS-x)/norm(x); % relative error 
    error_overall_MMVIAS = [error_overall_MMVIAS, error]; % store relative error 
    
    % overall error and SER for GSBL 
    error = norm(x_GSBL-x)/norm(x); % relative error 
    error_overall_GSBL = [error_overall_GSBL, error]; % store relative error  
    
    % overall error and SER for MMV-GSBL 
    error = norm(x_MMVGSBL-x)/norm(x); % relative error 
    error_overall_MMVGSBL = [error_overall_MMVGSBL, error]; % store relative error 

end 

%% Plot the results 

ms = 15; % marker size 
lw = 3.5; % line width 

% Plot relative coil errors for LS, IAS, and MMV-IAS  
fig1 = figure(1);
semilogy(lines,error_coil_LS,':o','Color','r','markersize',ms,'LineWidth',lw);
hold on
semilogy(lines,error_coil_IAS,'-^','Color','b','markersize',ms,'LineWidth',lw);
semilogy(lines,error_coil_MMVIAS,'--s','Color',[0 .6 0],'markersize',ms,'LineWidth',lw);
ax = gca;
ax.YMinorTick = 'off';
ax.YMinorGrid = 'off';
ax.FontSize = 26;
ax.LineWidth = 2;
ax.YGrid = 'on';
ax.XGrid = 'on';
ax.XTick = [20,40,60,80,100];
ax.YTick = [1e-2 1e-1 1e-0];
xlim([lines(1) lines(end)]);
ylim([4*1e-3 1.25*1e-0]);
h = legend('LS','IAS','MMV-IAS','location','best');
set(h,'Interpreter','latex','color','none','fontsize',26);
xlabel('#lines','interpreter','latex','fontsize',26);
ylabel('coil error','interpreter','latex','fontsize',26);
hold off

% Plot relative coil errors for LS, GSBL, and MMV-GSBL  
fig2 = figure(2);
semilogy(lines,error_coil_LS,':o','Color','r','markersize',ms,'LineWidth',lw);
hold on
semilogy(lines,error_coil_GSBL,'-^','Color','b','markersize',ms,'LineWidth',lw);
semilogy(lines,error_coil_MMVGSBL,'--s','Color',[0 .6 0],'markersize',ms,'LineWidth',lw);
ax = gca;
ax.YMinorTick = 'off';
ax.YMinorGrid = 'off';
ax.FontSize = 26;
ax.LineWidth = 2;
ax.YGrid = 'on';
ax.XGrid = 'on';
ax.XTick = [20,40,60,80,100];
ax.YTick = [1e-2 1e-1 1e-0];
xlim([lines(1) lines(end)]);
ylim([4*1e-3 1.25*1e-0]);
h = legend('LS','GSBL','MMV-GSBL','location','best');
set(h,'Interpreter','latex','color','none','fontsize',26);
xlabel('#lines','interpreter','latex','fontsize',26);
ylabel('coil error','interpreter','latex','fontsize',26);
hold off

% Plot overall errors for LS, IAS, and MMV-IAS  
fig3 = figure(3);
semilogy(lines,error_overall_LS,':o','Color','r','markersize',ms,'LineWidth',lw);
hold on
semilogy(lines,error_overall_IAS,'-^','Color','b','markersize',ms,'LineWidth',lw);
semilogy(lines,error_overall_MMVIAS,'--s','Color',[0 .6 0],'markersize',ms,'LineWidth',lw);
ax = gca;
ax.YMinorTick = 'off';
ax.YMinorGrid = 'off';
ax.FontSize = 26;
ax.LineWidth = 2;
ax.YGrid = 'on';
ax.XGrid = 'on';
ax.XTick = [20,40,60,80,100];
ax.YTick = [1e-2 1e-1 1e-0];
xlim([lines(1) lines(end)]);
ylim([4*1e-3 1.25*1e-0]);
h = legend('LS','IAS','MMV-IAS','location','best');
set(h,'Interpreter','latex','color','none','fontsize',26);
xlabel('lines','interpreter','latex','fontsize',26);
ylabel('overall error','interpreter','latex','fontsize',26);
hold off

% Plot overall errors for LS, GSBL, and MMV-GSBL  
fig4 = figure(4);
semilogy(lines,error_overall_LS,':o','Color','r','markersize',ms,'LineWidth',lw);
hold on
semilogy(lines,error_overall_GSBL,'-^','Color','b','markersize',ms,'LineWidth',lw);
semilogy(lines,error_overall_MMVGSBL,'--s','Color',[0 .6 0],'markersize',ms,'LineWidth',lw);
ax = gca;
ax.YMinorTick = 'off';
ax.YMinorGrid = 'off';
ax.FontSize = 26;
ax.LineWidth = 2;
ax.YGrid = 'on';
ax.XGrid = 'on';
ax.XTick = [20,40,60,80,100];
ax.YTick = [1e-2 1e-1 1e-0];
xlim([lines(1) lines(end)]);
ylim([4*1e-3 1.25*1e-0]);
h = legend('LS','GSBL','MMV-GSBL','location','best');
set(h,'Interpreter','latex','color','none','fontsize',26);
xlabel('lines','interpreter','latex','fontsize',26);
ylabel('overall error','interpreter','latex','fontsize',26);
hold off
