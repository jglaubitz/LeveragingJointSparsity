%% script_deconvolution_1d
%
% Description: 
%  Script to reconstruct a piecewise constant signal based on noisy blurred data. 
%
% Author: Jan Glaubitz 
% Date: Mar 27, 2023 
%

clear all; close all; clc; % clean up
%warning('off','all') % in case any of the warnings become too anoying 


%% Free parameters 

% Free parameters of the problem 
N = 40; % number of (equidistant) data points on [0,1] 
gamma = 3*10^(-2); % blurring parameter (Gaussian convolution kernel)
noise_variance = 1*10^(-2); % variance of the iid Gaussian noise added to the measurements
J = 4;
rng('default'); rng(1,'twister'); % to make the results reproducable

% Free parameters of MMV-GSBL 
c = 1; d = 10^(-4); 
% Free parameters of MMV-IAS 
%r = 1; beta_IAS = J/2 + 1 + 10^(-1); vartheta = 10^(-2); 
r = -1; beta_IAS = 1; vartheta = 10^(-4); 

%% Set up the model 

% Test function 
test_fun = @(t) (t<0.15).*(-1) + (t>=0.15 & t<0.25 ).*(0) + ... 
    (t>=0.25 & t<0.5 ).*(1) + (t>=0.5 & t<0.75 ).*(-0.5) + ... 
    (t>=0.75 & t<0.85 ).*(1.75) + (t>=0.85).*(0.5);

% Data points and signal values 
data_points = linspace(0, 1, N)'; % equidistant data points 
signal_values = rand(6, J); 

% forward operator, noise, and data 
F_conv = construct_F_deconvolution( N, gamma );  
for j=1:J 
    signal_values(:,j) = (2/max(signal_values(:,j)))*signal_values(:,j) - 1; % scale the values 
    fun{j} = @(t) (t<0.15).*signal_values(1,j) + ... 
        (t>=0.15 & t<0.25 ).*signal_values(2,j) + ... 
        (t>=0.25 & t<0.5 ).*signal_values(3,j) + ... 
        (t>=0.5 & t<0.75 ).*signal_values(4,j) + ... 
        (t>=0.75 & t<0.85 ).*signal_values(5,j) + ... 
        (t>=0.85).*signal_values(6,j);
    x{j} = fun{j}(data_points); % function values at grid points 
    noise_variances{j} = noise_variance; 
    F{j} = F_conv;  
    noise{j} = sqrt(noise_variance)*randn(size(F{j},1),1); % iid normal noise
    y{j} = F{j}*x{j} + noise{j}; % noisy indirect data 
end  

% Regularization operator 
order = 1; 
R = TV_operator( N, order ); % regularization operator 


%% Separate reconstructions using GSBL 
 
parfor j=1:J 
    % Use IAS and GSBL 
    [x_IAS{j}, theta_IAS{j}, history] = IAS_1d( F{j}, y{j}, noise_variance, R, r, beta_IAS, vartheta, 0 ); 
    [x_GSBL{j}, theta_GSBL{j}, history] = GSBL_1d( F{j}, y{j}, noise_variance, R, c, d, 0 ); % reconstruction based on j-th measurmeent y^(j) 
end 


%% Use MMV-IAS and -GSBL 
[x_MMVIAS, theta_MMVIAS, history] = MMV_IAS_1d( J, F, y, noise_variances, R, r, beta_IAS, vartheta, 0 );
[x_MMVGSBL, theta_MMVGSBL, history] = MMV_GSBL_1d( J, F, y, noise_variances, R, c, d, 0 );


%% Plot the results 

% Exact solution and measurements - 1st signal
figure(1) 
p1 = fplot( fun{1}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, y{1}, 'ro' ); 
set(p1, 'LineWidth',4);
set(p2, 'markersize',14, 'LineWidth',3); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1]; 
ylabel('$x$','Interpreter','latex'); 
grid on 
lgnd = legend( 'true signal $x$', 'noisy blurred data $\mathbf{y}$' );   
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northwest' )
hold off

% Exact solution and measurements - 2nd signal 
figure(2) 
p1 = fplot( fun{2}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, y{2}, 'ro' ); 
set(p1, 'LineWidth',4);
set(p2, 'markersize',14, 'LineWidth',3); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex'); 
grid on 
lgnd = legend( 'true signal $x$', 'noisy blurred data $\mathbf{y}$' );   
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northwest' )
hold off

% Reconstruction by IAS and MMV-IAS - 1st signal 
figure(3) 
p1 = fplot( fun{1}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, x_IAS{1}, 'b^', data_points, x_MMVIAS{1}, 'gs' );%, data_points, x_MMVGSBL{1}, 'rs'); % reconstructions 
set(p2(2), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'markersize',14, 'LineWidth',3); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northwest' )
hold off 

% Reconstruction by IAS and MMV-IAS - 2nd signal
figure(4) 
p1 = fplot( fun{2}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, x_IAS{2}, 'b^', data_points, x_MMVIAS{2}, 'gs' );%, data_points, x_MMVGSBL{2}, 'rs'); % reconstructions 
set(p2(2), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'markersize',14, 'LineWidth',3); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northwest' )
hold off

% Reconstruction by GSBL and MMV-GSBL - 1st signal
figure(5) 
p1 = fplot( fun{1}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, x_GSBL{1}, 'b^', data_points, x_MMVGSBL{1}, 'gs' );%, data_points, x_MMVGSBL{1}, 'rs'); % reconstructions 
set(p2(2), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'markersize',14, 'LineWidth',3); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'GSBL', 'MMV-GSBL');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northwest' )
hold off 

% Reconstruction by GSBL and MMV-GSBL - 2nd signal 
figure(6) 
p1 = fplot( fun{2}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, x_GSBL{2}, 'b^', data_points, x_MMVGSBL{2}, 'gs' );%, data_points, x_MMVGSBL{2}, 'rs'); % reconstructions 
set(p2(2), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'markersize',14, 'LineWidth',3); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'GSBL', 'MMV-GSBL');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northwest' )
hold off 

% Hyper-parameters for IAS and MMV-IAS - 1st signal
t = ( data_points(1:end-1) + data_points(2:end) )/2;
figure(7) 
p1 = fplot( fun{1}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( t, theta_IAS{1}/max(theta_IAS{1}), 'b-', t, theta_MMVIAS/max(theta_MMVIAS), 'g--' ); 
set(p2(2), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'LineWidth',4); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$\theta$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northwest' )
hold off 

% Hyper-parameters for IAS and MMV-IAS - 2nd signal
figure(8) 
p1 = fplot( fun{2}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( t, theta_IAS{2}/max(theta_IAS{2}), 'b-', t, theta_MMVIAS/max(theta_MMVIAS), 'g--' ); 
set(p2(2), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'LineWidth',4); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$\theta$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northwest' )
hold off 

% Hyper-parameters for GSBL and MMV-GSBL - 1st signal
figure(9) 
p1 = fplot( fun{1}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( t, (1./theta_GSBL{1})/max((1./theta_GSBL{1})), 'b-', t, (1./theta_MMVGSBL)/max((1./theta_MMVGSBL)), 'g--' ); 
set(p2(2), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'LineWidth',4); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$\theta^{-1}$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'GSBL', 'MMV-GSBL');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northwest' )
hold off 

% Hyper-parameters for GSBL and MMV-GSBL - 2nd signal
figure(10) 
p1 = fplot( fun{2}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( t, (1./theta_GSBL{2})/max((1./theta_GSBL{2})), 'b-', t, (1./theta_MMVGSBL)/max((1./theta_MMVGSBL)), 'g--' ); 
set(p2(2), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'LineWidth',4); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$\theta^{-1}$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'GSBL', 'MMV-GSBL');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northwest' )
hold off 

% Credible intervals for IAS - 1st signal 
Gamma_inv = (1/noise_variances{1})*F{1}'*F{1} + R'*diag(theta_IAS{1}.^(-1))*R; % precision matrix
Gamma = inv( Gamma_inv ); % get the covariance matrix 
[CI_lower, CI_upper] = compute_CI( x_IAS{1}, Gamma ); % computer lower and upper bounds
figure(11) 
p1 = fplot( fun{1}, [0,1], 'k:'); % plot the reference solution 
set(p1, 'LineWidth',3); 
hold on 
p2 = plot( data_points, x_IAS{1}, 'gs' ); % reconstructions 
set(p2(1), 'color', [0 0.6 0]); 
set(p2, 'markersize',14, 'LineWidth',3);  
p3 = patch([data_points; flipud(data_points)], [CI_lower; flipud(CI_upper)], [1 0.4 0]);  
set(p3, 'EdgeColor','none', 'FaceAlpha',0.35); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex');  
grid on 
lgnd = legend([p2 p3], 'mean (IAS)','$99.9\%$ CI (IAS)');
set(lgnd, 'Interpreter','latex', 'Location','northeast', 'FontSize',26, 'color','none');
hold off 

% Credible intervals for MMV-IAS - 1st signal 
Gamma_inv = (1/noise_variances{1})*F{1}'*F{1} + R'*diag(theta_MMVIAS.^(-1))*R; % precision matrix
Gamma = inv( Gamma_inv ); % get the covariance matrix 
[CI_lower, CI_upper] = compute_CI( x_MMVIAS{1}, Gamma ); % computer lower and upper bounds
figure(12) 
p1 = fplot( fun{1}, [0,1], 'k:'); % plot the reference solution 
set(p1, 'LineWidth',3); 
hold on 
p2 = plot( data_points, x_MMVIAS{1}, 'gs' ); % reconstructions 
set(p2(1), 'color', [0 0.6 0]); 
set(p2, 'markersize',14, 'LineWidth',3);  
p3 = patch([data_points; flipud(data_points)], [CI_lower; flipud(CI_upper)], [1 0.4 0]);  
set(p3, 'EdgeColor','none', 'FaceAlpha',0.35); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex');  
grid on 
lgnd = legend([p2 p3], 'mean (MMV-IAS)','$99.9\%$ CI (MMV-IAS)');  
set(lgnd, 'Interpreter','latex', 'Location','northeast', 'FontSize',26, 'color','none');
hold off 

% Credible intervals for IAS - 2nd signal 
Gamma_inv = (1/noise_variances{2})*F{2}'*F{2} + R'*diag(theta_IAS{2}.^(-1))*R; % precision matrix
Gamma = inv( Gamma_inv ); % get the covariance matrix 
[CI_lower, CI_upper] = compute_CI( x_IAS{2}, Gamma ); % computer lower and upper bounds
figure(13) 
p1 = fplot( fun{2}, [0,1], 'k:'); % plot the reference solution 
set(p1, 'LineWidth',3); 
hold on 
p2 = plot( data_points, x_IAS{2}, 'gs' ); % reconstructions 
set(p2(1), 'color', [0 0.6 0]); 
set(p2, 'markersize',14, 'LineWidth',3);  
p3 = patch([data_points; flipud(data_points)], [CI_lower; flipud(CI_upper)], [1 0.4 0]);  
set(p3, 'EdgeColor','none', 'FaceAlpha',0.35); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex');  
grid on 
lgnd = legend([p2 p3], 'mean (IAS)','$99.9\%$ CI (IAS)');  
set(lgnd, 'Interpreter','latex', 'Location','northwest', 'FontSize',26, 'color','none');
hold off 

% Credible intervals for MMV-IAS - 2nd signal 
Gamma_inv = (1/noise_variances{2})*F{2}'*F{2} + R'*diag(theta_MMVIAS.^(-1))*R; % precision matrix
Gamma = inv( Gamma_inv ); % get the covariance matrix 
[CI_lower, CI_upper] = compute_CI( x_MMVIAS{2}, Gamma ); % computer lower and upper bounds
figure(14) 
p1 = fplot( fun{2}, [0,1], 'k:'); % plot the reference solution 
set(p1, 'LineWidth',3); 
hold on 
p2 = plot( data_points, x_MMVIAS{2}, 'gs' ); % reconstructions 
set(p2(1), 'color', [0 0.6 0]); 
set(p2, 'markersize',14, 'LineWidth',3);  
p3 = patch([data_points; flipud(data_points)], [CI_lower; flipud(CI_upper)], [1 0.4 0]);  
set(p3, 'EdgeColor','none', 'FaceAlpha',0.35); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex');  
grid on 
lgnd = legend([p2 p3], 'mean (MMV-IAS)','$99.9\%$ CI (MMV-IAS)');  
set(lgnd, 'Interpreter','latex', 'Location','northwest', 'FontSize',26, 'color','none');
hold off 

% Credible intervals for GSBL - 1st signal 
Gamma_inv = (1/noise_variances{1})*F{1}'*F{1} + R'*diag(theta_GSBL{1})*R; % precision matrix
Gamma = inv( Gamma_inv ); % get the covariance matrix 
[CI_lower, CI_upper] = compute_CI( x_GSBL{1}, Gamma ); % computer lower and upper bounds
figure(15) 
p1 = fplot( fun{1}, [0,1], 'k:'); % plot the reference solution 
set(p1, 'LineWidth',3); 
hold on 
p2 = plot( data_points, x_GSBL{1}, 'gs' ); % reconstructions 
set(p2(1), 'color', [0 0.6 0]); 
set(p2, 'markersize',14, 'LineWidth',3);  
p3 = patch([data_points; flipud(data_points)], [CI_lower; flipud(CI_upper)], [1 0.4 0]);  
set(p3, 'EdgeColor','none', 'FaceAlpha',0.35); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex');  
grid on 
lgnd = legend([p2 p3], 'mean (GSBL)','$99.9\%$ CI (GSBL)');  
set(lgnd, 'Interpreter','latex', 'Location','northeast', 'FontSize',26, 'color','none');
hold off 

% Credible intervals for MMV-GSBL - 1st signal 
Gamma_inv = (1/noise_variances{1})*F{1}'*F{1} + R'*diag(theta_MMVGSBL)*R; % precision matrix
Gamma = inv( Gamma_inv ); % get the covariance matrix 
[CI_lower, CI_upper] = compute_CI( x_MMVGSBL{1}, Gamma ); % computer lower and upper bounds
figure(16) 
p1 = fplot( fun{1}, [0,1], 'k:'); % plot the reference solution 
set(p1, 'LineWidth',3); 
hold on 
p2 = plot( data_points, x_MMVGSBL{1}, 'gs' ); % reconstructions 
set(p2(1), 'color', [0 0.6 0]); 
set(p2, 'markersize',14, 'LineWidth',3);  
p3 = patch([data_points; flipud(data_points)], [CI_lower; flipud(CI_upper)], [1 0.4 0]);  
set(p3, 'EdgeColor','none', 'FaceAlpha',0.35); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex');  
grid on 
lgnd = legend([p2 p3], 'mean (MMV-GSBL)','$99.9\%$ CI (MMV-GSBL)');  
set(lgnd, 'Interpreter','latex', 'Location','northeast', 'FontSize',24, 'color','none');
hold off 

% Credible intervals for GSBL - 2nd signal 
Gamma_inv = (1/noise_variances{2})*F{2}'*F{2} + R'*diag(theta_GSBL{2})*R; % precision matrix
Gamma = inv( Gamma_inv ); % get the covariance matrix 
[CI_lower, CI_upper] = compute_CI( x_GSBL{2}, Gamma ); % computer lower and upper bounds
figure(17) 
p1 = fplot( fun{2}, [0,1], 'k:'); % plot the reference solution 
set(p1, 'LineWidth',3); 
hold on 
p2 = plot( data_points, x_GSBL{2}, 'gs' ); % reconstructions 
set(p2(1), 'color', [0 0.6 0]); 
set(p2, 'markersize',14, 'LineWidth',3);  
p3 = patch([data_points; flipud(data_points)], [CI_lower; flipud(CI_upper)], [1 0.4 0]);  
set(p3, 'EdgeColor','none', 'FaceAlpha',0.35); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex');  
grid on 
lgnd = legend([p2 p3], 'mean (GSBL)','$99.9\%$ CI (GSBL)');  
set(lgnd, 'Interpreter','latex', 'Location','northwest', 'FontSize',26, 'color','none');
hold off 

% Credible intervals for MMV-GSBL - 2nd signal 
Gamma_inv = (1/noise_variances{2})*F{2}'*F{2} + R'*diag(theta_MMVGSBL)*R; % precision matrix
Gamma = inv( Gamma_inv ); % get the covariance matrix 
[CI_lower, CI_upper] = compute_CI( x_MMVGSBL{2}, Gamma ); % computer lower and upper bounds
figure(18) 
p1 = fplot( fun{2}, [0,1], 'k:'); % plot the reference solution 
set(p1, 'LineWidth',3); 
hold on 
p2 = plot( data_points, x_MMVGSBL{2}, 'gs' ); % reconstructions 
set(p2(1), 'color', [0 0.6 0]); 
set(p2, 'markersize',14, 'LineWidth',3);  
p3 = patch([data_points; flipud(data_points)], [CI_lower; flipud(CI_upper)], [1 0.4 0]);  
set(p3, 'EdgeColor','none', 'FaceAlpha',0.35); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex');  
grid on 
lgnd = legend([p2 p3], 'mean (MMV-GSBL)','$99.9\%$ CI (MMV-GSBL)');  
set(lgnd, 'Interpreter','latex', 'Location','northwest', 'FontSize',26, 'color','none');
hold off 