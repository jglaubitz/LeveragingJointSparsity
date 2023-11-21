%% script_comparison
%
% Description: 
%  Script to compare the MMV approach to using previsously learned hyper-parameters 
%  as an initialization for recovering the next signal  
%
% Author: Jan Glaubitz 
% Date: Nov 09, 2023 
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
r1 = 1; beta1_IAS = 3/2 + 10^(-3); vartheta1_IAS = 10^(-2); 
beta1_MMVIAS = 1 + J/2 + 10^(-3); vartheta1_MMVIAS = 10^(-2); 
r2 = -1; beta2_IAS = 1; vartheta2_IAS = 10^(-4); 
beta2_MMVIAS = 1; vartheta2_MMVIAS = 10^(-4); 

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


%% Separate reconstructions using IAS and GSBL 
parfor j=1:J 

    % Use IAS and GSBL 
    [x_IAS_rp1{j}, theta_IAS_rp1{j}, history] = IAS_1d( F{j}, y{j}, noise_variance, R, r1, beta1_IAS, vartheta1_IAS, 1 ); 
    [x_IAS_rm1{j}, theta_IAS_rm1{j}, history] = IAS_1d( F{j}, y{j}, noise_variance, R, r2, beta2_IAS, vartheta2_IAS, 1 ); 
    [x_GSBL{j}, theta_GSBL{j}, history] = GSBL_1d( F{j}, y{j}, noise_variance, R, c, d, 1 ); 

end 


%% Separate reconstructions using sequantial IAS and GSBL 
for j=1:J 
    
    % Get the initial values for the theta-vector 
    if j == 1 % for first image initialization is the same as in IAS/GSBL algorithm  
        K = size(R,1); % number of outputs of the regularization operator 
        theta_IAS_rp1_init = ones(K,1); 
        theta_IAS_rm1_init = ones(K,1);
        theta_GSBL_init = ones(K,1);
    else % otherwise theta is initialized by the output of the previsous reconstruction 
        theta_IAS_rp1_init = theta_IAS_rp1_seq{j-1}; 
        theta_IAS_rm1_init = theta_IAS_rm1_seq{j-1};
        theta_GSBL_init = theta_GSBL_seq{j-1};
    end

    % Compute the next reconstruction 
    [x_IAS_rp1_seq{j}, theta_IAS_rp1_seq{j}, history] = seq_IAS_1d( F{j}, y{j}, noise_variance, R, r1, beta1_IAS, vartheta1_IAS, theta_IAS_rp1_init, 1 ); 
    [x_IAS_rm1_seq{j}, theta_IAS_rm1_seq{j}, history] = seq_IAS_1d( F{j}, y{j}, noise_variance, R, r2, beta2_IAS, vartheta2_IAS, theta_IAS_rm1_init, 1 ); 
    [x_GSBL_seq{j}, theta_GSBL_seq{j}, history] = seq_GSBL_1d( F{j}, y{j}, noise_variance, R, c, d, theta_GSBL_init, 1 ); 

end 


%% Use MMV-IAS and -GSBL 
[x_MMVIAS_rp1, theta_MMVIAS_rp1, history] = MMV_IAS_1d( J, F, y, noise_variances, R, r1, beta1_MMVIAS, vartheta1_MMVIAS, 1 );
[x_MMVIAS_rm1, theta_MMVIAS_rm1, history] = MMV_IAS_1d( J, F, y, noise_variances, R, r2, beta2_MMVIAS, vartheta2_MMVIAS, 1 );
[x_MMVGSBL, theta_MMVGSBL, history] = MMV_GSBL_1d( J, F, y, noise_variances, R, c, d, 1 );


%% Plot the results 

% Exact solution and measurements - 1st signal
fig = figure(1) 
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
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northeast' )
hold off
%saveas(fig, 'comp_signal1.eps', 'epsc');

% Exact solution and measurements - 2nd signal 
fig = figure(2) 
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
%saveas(fig, 'comp_signal2.eps', 'epsc');

% Exact solution and measurements - 3rd signal 
fig = figure(3) 
p1 = fplot( fun{3}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, y{3}, 'ro' ); 
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
%saveas(fig, 'comp_signal3.eps', 'epsc');

% Exact solution and measurements - 4th signal 
fig = figure(4) 
p1 = fplot( fun{4}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, y{4}, 'ro' ); 
set(p1, 'LineWidth',4);
set(p2, 'markersize',14, 'LineWidth',3); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex'); 
grid on 
lgnd = legend( 'true signal $x$', 'noisy blurred data $\mathbf{y}$' );   
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'southeast' )
hold off
%saveas(fig, 'comp_signal4.eps', 'epsc');


% Reconstruction by IAS and MMV-IAS (r=1) - 1st signal 
fig = figure(5) 
p1 = fplot( fun{1}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, x_IAS_rp1{1}, 'b^', data_points, x_IAS_rp1_seq{1}, 'r*' , data_points, x_MMVIAS_rp1{1}, 'gs' ); % reconstructions 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'markersize',14, 'LineWidth',3); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'seq. IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northeast' )
hold off 
%saveas(fig, 'comp_signal1_IAS_rp1.eps', 'epsc');

% Reconstruction by IAS and MMV-IAS (r=1) - 2nd signal 
fig = figure(6) 
p1 = fplot( fun{2}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, x_IAS_rp1{2}, 'b^', data_points, x_IAS_rp1_seq{2}, 'r*' , data_points, x_MMVIAS_rp1{2}, 'gs' ); % reconstructions 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'markersize',14, 'LineWidth',3); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'seq. IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northwest' )
hold off 
%saveas(fig, 'comp_signal2_IAS_rp1.eps', 'epsc');

% Reconstruction by IAS and MMV-IAS (r=1) - 3rd signal 
fig = figure(7) 
p1 = fplot( fun{3}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, x_IAS_rp1{3}, 'b^', data_points, x_IAS_rp1_seq{3}, 'r*' , data_points, x_MMVIAS_rp1{3}, 'gs' ); % reconstructions 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'markersize',14, 'LineWidth',3); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'seq. IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'southeast' )
hold off 
%saveas(fig, 'comp_signal3_IAS_rp1.eps', 'epsc');

% Reconstruction by IAS and MMV-IAS (r=1) - 4th signal 
fig = figure(8) 
p1 = fplot( fun{4}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, x_IAS_rp1{4}, 'b^', data_points, x_IAS_rp1_seq{4}, 'r*' , data_points, x_MMVIAS_rp1{4}, 'gs' ); % reconstructions 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'markersize',14, 'LineWidth',3); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'seq. IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'south' )
hold off 
%saveas(fig, 'comp_signal4_IAS_rp1.eps', 'epsc');


% Reconstruction by IAS and MMV-IAS (r=-1) - 1st signal 
fig = figure(9) 
p1 = fplot( fun{1}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, x_IAS_rm1{1}, 'b^', data_points, x_IAS_rm1_seq{1}, 'r*' , data_points, x_MMVIAS_rm1{1}, 'gs' ); % reconstructions 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'markersize',14, 'LineWidth',3); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'seq. IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northeast' )
hold off 
%saveas(fig, 'comp_signal1_IAS_rm1.eps', 'epsc');

% Reconstruction by IAS and MMV-IAS (r=-1) - 2nd signal 
fig = figure(10) 
p1 = fplot( fun{2}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, x_IAS_rm1{2}, 'b^', data_points, x_IAS_rm1_seq{2}, 'r*' , data_points, x_MMVIAS_rm1{2}, 'gs' ); % reconstructions 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'markersize',14, 'LineWidth',3); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'seq. IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northwest' )
hold off 
%saveas(fig, 'comp_signal2_IAS_rm1.eps', 'epsc');

% Reconstruction by IAS and MMV-IAS (r=-1) - 3rd signal 
fig = figure(11) 
p1 = fplot( fun{3}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, x_IAS_rm1{3}, 'b^', data_points, x_IAS_rm1_seq{3}, 'r*' , data_points, x_MMVIAS_rm1{3}, 'gs' ); % reconstructions 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'markersize',14, 'LineWidth',3); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'seq. IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'southeast' )
hold off 
%saveas(fig, 'comp_signal3_IAS_rm1.eps', 'epsc');

% Reconstruction by IAS and MMV-IAS (r=-1) - 4th signal 
fig = figure(12) 
p1 = fplot( fun{4}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, x_IAS_rm1{4}, 'b^', data_points, x_IAS_rm1_seq{4}, 'r*' , data_points, x_MMVIAS_rm1{4}, 'gs' ); % reconstructions 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'markersize',14, 'LineWidth',3); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'seq. IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'south' )
hold off 
%saveas(fig, 'comp_signal4_IAS_rm1.eps', 'epsc');


% Reconstruction by GSBL and MMV-GSBL - 1st signal 
fig = figure(13) 
p1 = fplot( fun{1}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, x_GSBL{1}, 'b^', data_points, x_GSBL_seq{1}, 'r*' , data_points, x_MMVGSBL{1}, 'gs' ); % reconstructions 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'markersize',14, 'LineWidth',3); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'GSBL', 'seq. GSBL', 'MMV-GSBL');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northeast' )
hold off 
%saveas(fig, 'comp_signal1_GSBL.eps', 'epsc');

% Reconstruction by GSBL and MMV-GSBL - 2nd signal 
fig = figure(14) 
p1 = fplot( fun{2}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, x_GSBL{2}, 'b^', data_points, x_GSBL_seq{2}, 'r*' , data_points, x_MMVGSBL{2}, 'gs' ); % reconstructions 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'markersize',14, 'LineWidth',3); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'GSBL', 'seq. GSBL', 'MMV-GSBL');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northwest' )
hold off 
%saveas(fig, 'comp_signal2_GSBL.eps', 'epsc');

% Reconstruction by GSBL and MMV-GSBL - 3rd signal 
fig = figure(15) 
p1 = fplot( fun{3}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, x_GSBL{3}, 'b^', data_points, x_GSBL_seq{3}, 'r*' , data_points, x_MMVGSBL{3}, 'gs' ); % reconstructions 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'markersize',14, 'LineWidth',3); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'GSBL', 'seq. GSBL', 'MMV-GSBL');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'southeast' )
hold off 
%saveas(fig, 'comp_signal3_GSBL.eps', 'epsc');

% Reconstruction by GSBL and MMV-GSBL - 4th signal 
fig = figure(16) 
p1 = fplot( fun{4}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, x_GSBL{4}, 'b^', data_points, x_GSBL_seq{4}, 'r*' , data_points, x_MMVGSBL{4}, 'gs' ); % reconstructions 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'markersize',14, 'LineWidth',3); 
set(gca, 'FontSize', 26); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$x$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'GSBL', 'seq. GSBL', 'MMV-GSBL');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'south' )
hold off 
%saveas(fig, 'comp_signal4_GSBL.eps', 'epsc');


% Hyper-parameters for IAS and MMV-IAS (r=1) - 1st signal
t = ( data_points(1:end-1) + data_points(2:end) )/2;
fig = figure(17) 
p1 = fplot( fun{1}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( t, theta_IAS_rp1{1}/max(theta_IAS_rp1{1}), 'b-', t, theta_IAS_rp1_seq{1}/max(theta_IAS_rp1_seq{1}), 'r-.' , t, theta_MMVIAS_rp1/max(theta_MMVIAS_rp1), 'g--' ); 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'LineWidth',4); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$\theta$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'seq. IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northeast' )
hold off 
%saveas(fig, 'comp_signal1_IAS_rp1_theta.eps', 'epsc');

% Hyper-parameters for IAS and MMV-IAS (r=1) - 2nd signal
t = ( data_points(1:end-1) + data_points(2:end) )/2;
fig = figure(18) 
p1 = fplot( fun{1}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( t, theta_IAS_rp1{2}/max(theta_IAS_rp1{2}), 'b-', t, theta_IAS_rp1_seq{2}/max(theta_IAS_rp1_seq{2}), 'r-.' , t, theta_MMVIAS_rp1/max(theta_MMVIAS_rp1), 'g--' ); 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'LineWidth',4); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$\theta$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'seq. IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'south' )
hold off 
%saveas(fig, 'comp_signal2_IAS_rp1_theta.eps', 'epsc');

% Hyper-parameters for IAS and MMV-IAS (r=1) - 3rd signal
t = ( data_points(1:end-1) + data_points(2:end) )/2;
fig = figure(19) 
p1 = fplot( fun{3}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( t, theta_IAS_rp1{3}/max(theta_IAS_rp1{3}), 'b-', t, theta_IAS_rp1_seq{3}/max(theta_IAS_rp1_seq{3}), 'r-.' , t, theta_MMVIAS_rp1/max(theta_MMVIAS_rp1), 'g--' ); 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'LineWidth',4); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$\theta$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'seq. IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'southeast' )
hold off 
%saveas(fig, 'comp_signal3_IAS_rp1_theta.eps', 'epsc');

% Hyper-parameters for IAS and MMV-IAS (r=1) - 4th signal
t = ( data_points(1:end-1) + data_points(2:end) )/2;
fig = figure(20) 
p1 = fplot( fun{1}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( t, theta_IAS_rp1{4}/max(theta_IAS_rp1{4}), 'b-', t, theta_IAS_rp1_seq{4}/max(theta_IAS_rp1_seq{4}), 'r-.' , t, theta_MMVIAS_rp1/max(theta_MMVIAS_rp1), 'g--' ); 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'LineWidth',4); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$\theta$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'seq. IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'south' )
hold off 
%saveas(fig, 'comp_signal4_IAS_rp1_theta.eps', 'epsc');


% Hyper-parameters for IAS and MMV-IAS (r=-1) - 1st signal
t = ( data_points(1:end-1) + data_points(2:end) )/2;
fig = figure(21) 
p1 = fplot( fun{1}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( t, theta_IAS_rm1{1}/max(theta_IAS_rm1{1}), 'b-', t, theta_IAS_rm1_seq{1}/max(theta_IAS_rm1_seq{1}), 'r-.' , t, theta_MMVIAS_rm1/max(theta_MMVIAS_rm1), 'g--' ); 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'LineWidth',4); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$\theta$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'seq IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northeast' )
hold off 
%saveas(fig, 'comp_signal1_IAS_rm1_theta.eps', 'epsc');

% Hyper-parameters for IAS and MMV-IAS (r=-1) - 2nd signal
t = ( data_points(1:end-1) + data_points(2:end) )/2;
fig = figure(22) 
p1 = fplot( fun{1}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( t, theta_IAS_rm1{2}/max(theta_IAS_rm1{2}), 'b-', t, theta_IAS_rm1_seq{2}/max(theta_IAS_rm1_seq{2}), 'r-.' , t, theta_MMVIAS_rm1/max(theta_MMVIAS_rm1), 'g--' ); 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'LineWidth',4); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$\theta$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'seq. IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'south' )
hold off 
%saveas(fig, 'comp_signal2_IAS_rm1_theta.eps', 'epsc');

% Hyper-parameters for IAS and MMV-IAS (r=-1) - 3rd signal
t = ( data_points(1:end-1) + data_points(2:end) )/2;
fig = figure(23) 
p1 = fplot( fun{3}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( t, theta_IAS_rm1{3}/max(theta_IAS_rm1{3}), 'b-', t, theta_IAS_rm1_seq{3}/max(theta_IAS_rm1_seq{3}), 'r-.' , t, theta_MMVIAS_rm1/max(theta_MMVIAS_rm1), 'g--' ); 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'LineWidth',4); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$\theta$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'seq. IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'southeast' )
hold off 
%saveas(fig, 'comp_signal3_IAS_rm1_theta.eps', 'epsc');

% Hyper-parameters for IAS and MMV-IAS (r=-1) - 4th signal
t = ( data_points(1:end-1) + data_points(2:end) )/2;
fig = figure(24) 
p1 = fplot( fun{1}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( t, theta_IAS_rm1{4}/max(theta_IAS_rm1{4}), 'b-', t, theta_IAS_rm1_seq{4}/max(theta_IAS_rm1_seq{4}), 'r-.' , t, theta_MMVIAS_rm1/max(theta_MMVIAS_rm1), 'g--' ); 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'LineWidth',4); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$\theta$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'IAS', 'seq. IAS', 'MMV-IAS');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'north' )
hold off 
%saveas(fig, 'comp_signal4_IAS_rm1_theta.eps', 'epsc');


% Hyper-parameters for GSBL and MMV-GSBL - 1st signal
t = ( data_points(1:end-1) + data_points(2:end) )/2;
fig = figure(25) 
p1 = fplot( fun{1}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( t, (1./theta_GSBL{1})/max((1./theta_GSBL{1})), 'b-', t, (1./theta_GSBL_seq{1})/max((1./theta_GSBL_seq{1})), 'r-.' , t, (1./theta_MMVGSBL)/max((1./theta_MMVGSBL)), 'g--' ); 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'LineWidth',4); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$\theta$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'GSBL', 'seq. GSBL', 'MMV-GSBL');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'northeast' )
hold off 
%saveas(fig, 'comp_signal1_GSBL_theta.eps', 'epsc');

% Hyper-parameters for GSBL and MMV-GSBL - 2nd signal
t = ( data_points(1:end-1) + data_points(2:end) )/2;
fig = figure(26) 
p1 = fplot( fun{2}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( t, (1./theta_GSBL{2})/max((1./theta_GSBL{2})), 'b-', t, (1./theta_GSBL_seq{2})/max((1./theta_GSBL_seq{2})), 'r-.' , t, (1./theta_MMVGSBL)/max((1./theta_MMVGSBL)), 'g--' ); 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'LineWidth',4); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$\theta$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'GSBL', 'seq. GSBL', 'MMV-GSBL');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'southeast' )
hold off 
%saveas(fig, 'comp_signal2_GSBL_theta.eps', 'epsc');

% Hyper-parameters for GSBL and MMV-GSBL - 3rd signal
t = ( data_points(1:end-1) + data_points(2:end) )/2;
fig = figure(27) 
p1 = fplot( fun{3}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( t, (1./theta_GSBL{3})/max((1./theta_GSBL{3})), 'b-', t, (1./theta_GSBL_seq{3})/max((1./theta_GSBL_seq{3})), 'r-.' , t, (1./theta_MMVGSBL)/max((1./theta_MMVGSBL)), 'g--' ); 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'LineWidth',4); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$\theta$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'GSBL', 'seq. GSBL', 'MMV-GSBL');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'southeast' )
hold off 
%saveas(fig, 'comp_signal3_GSBL_theta.eps', 'epsc');

% Hyper-parameters for GSBL and MMV-GSBL - 4th signal
t = ( data_points(1:end-1) + data_points(2:end) )/2;
fig = figure(28) 
p1 = fplot( fun{4}, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( t, (1./theta_GSBL{4})/max((1./theta_GSBL{4})), 'b-', t, (1./theta_GSBL_seq{4})/max((1./theta_GSBL_seq{4})), 'r-.' , t, (1./theta_MMVGSBL)/max((1./theta_MMVGSBL)), 'g--' ); 
set(p2(3), 'color', [0 0.6 0])
set(p1, 'LineWidth',4); 
set(p2, 'LineWidth',4); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
ax = gca;
ax.XTick = [0 .2 .4 .6 .8 1];  
ylabel('$\theta$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'GSBL', 'seq. GSBL', 'MMV-GSBL');
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location', 'south' )
hold off 
%saveas(fig, 'comp_signal4_GSBL_theta.eps', 'epsc');