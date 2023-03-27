%% script_pMRI
%
% Description: 
%  Script to perform multi-coil parallel magnetic resonance imaging (mPRI)
%
% Author: Jan Glaubitz 
% Date: Mar 27, 2023 
%

clear all; close all; clc; % clean up
%warning('off','all') % in case any of the warnings become too anoying 

%% Free parameters 

% Free parameters of the problem 
N = 256; % number ofpixels in each direction 
C = 4; % number of coils 
lines = 20; % number of angles
sigma_sq = 10^(-3); % noise variance
rng('default'); rng(1,'twister'); % to make the results reproducable

% Free parameters of the methods 
reg_type = 'TV'; % type of regualrization (TV = total variation, PA = polynomial annihilation) 
order = 1;
c_GSBL = 1; d_GSBL = 10^(-3); % Free parameters of GSBL 
r = -1; beta_IAS = 1; vartheta = 10^(-3); % Free parameters of IAS 


%% Construct coils and images 

% Set up the brain image  
X = phantom('Modified Shepp-Logan',N);
RI = imref2d(size(X)); 
x = reshape(X,N*N,1); % vectorize the reference image 

figure(1)
tlo = tiledlayout(1,1,'TileSpacing','none','Padding','none');
nexttile, imshow(X); 
axis square;
colormap(gray)

xCrecLS = zeros(N*N,C); % recovered vectorized coil images using LS
xCrecIAS = zeros(N*N,C); % recovered vectorized coil images using IAS
xCrecMMVIAS = zeros(N*N,C); % recovered vectorized coil images using MMV-IAS
xCrecGSBL = zeros(N*N,C);% recovered vectorized coil images using GSBL
xCrecMMVGSBL = zeros(N*N,C); % recovered vectorized coil images using MMV-GSBL

error_coil_IAS = []; % coil image error IAS
error_overall_IAS = []; % overall image error IAS
error_coil_MMVIAS = []; % coil image error IAS
error_overall_MMVIAS = []; % overall image error MMV-IAS
error_coil_GSBL = []; % coil image error GSBL
error_overall_GSBL = []; % overall image error MMV-IAS 
error_coil_MMVGSBL = []; % coil image error MMV-GSBL
error_overall_MMVGSBL = []; % overall image error MMV-IAS

l = lines;

% Construct vectorized coils, coil images, and measurements 
for c=1:C 
    % Construct sampling operator  
    [idx1{c},idx2{c},R{c}] = MMV_SampMatrixRadial(N,l,c,C);
    m{c} = nnz(R{c});
    opF{c} = @(x,mode) partialFourierShift2D(idx1{c},idx2{c},N,x,mode); % forward operator
    noise_var{c} = sigma_sq; % we use the same noise variance for all MMVs 
    noise{c} = sqrt(noise_var{c})*randn(m{c},1); % iid zero-mean normal noise 
    y{c} = opF{c}(x,'notransp') + noise{c}; % compute measurements 
end
% Construct function handle for the regularization operator
opR = @(x,flag) reg_op( N, x, reg_type, order, flag ); % regularization operator

% Illustrate sample distributions 
figure(2);
tlo = tiledlayout(1,4,'TileSpacing','none','Padding','none');
for c=1:4 
    % sensitivity profile
    %subplot(2,2,c)
    nexttile, imshow(R{c}, 'InitialMagnification',2000);
    colormap(gray)
    axis square;
    axis off;
    %title(['Samples c = ',num2str(c)],'fontsize',12);
end


%% Separate reconstructions using LS, IAS, and GSBL 

parfor c=1:C  
    xCrecLS(:,c) = lsqr( opF{c}, y{c} ); % least squares approximation    
    [xCrecIAS(:,c), theta, history] = IAS( opF{c}, y{c}, noise_var{c}, opR, r, beta_IAS, vartheta, 1 ); % Use IAS
    [xCrecGSBL(:,c), theta, history] = GSBL( opF{c}, y{c}, noise_var{c}, opR, c_GSBL, d_GSBL, 1 ); % Use GSBL
end 


%% Joint reconstruction using MMV-IAS and -GSBL 
[x_MMVIAS, theta, history] = MMV_IAS( C, opF, y, noise_var, opR, r, beta_IAS, vartheta, 0 );
[x_MMVGSBL, theta, history] = MMV_GSBL( C, opF, y, noise_var, opR, c_GSBL, d_GSBL, 0 );

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


%% Plot the results 
c=1; 

% Illustrate LS coil images 
figure(3);
%phplot(reshape(xCrecLS(:,c),N,N));
imshow(reshape(xCrecLS(:,c),N,N));
axis square;
axis off;
title(['LS, c = ',num2str(c)],'fontsize',12);

% Illustrate IAS coil images 
figure(4);
%phplot(reshape(xCrecIAS(:,c),N,N));
imshow(reshape(xCrecIAS(:,c),N,N));
axis square;
axis off;
title(['IAS, c = ',num2str(c)],'fontsize',12);

% Illustrate MMV-IAS coil images 
figure(5);
%phplot(reshape(xCrecMMVIAS(:,c),N,N));
imshow(reshape(xCrecMMVIAS(:,c),N,N));
axis square;
axis off;
title(['MMV-IAS, c = ',num2str(c)],'fontsize',12);

% Illustrate GSBL coil images 
figure(6);
%phplot(reshape(xCrecGSBL(:,c),N,N));
imshow(reshape(xCrecGSBL(:,c),N,N));
axis square;
axis off;
title(['GSBL, c = ',num2str(c)],'fontsize',12);

% Illustrate MMV-GSBL coil images 
figure(7);
%phplot(reshape(xCrecMMVGSBL(:,c),N,N));
imshow(reshape(xCrecMMVGSBL(:,c),N,N));
axis square;
axis off;
title(['MMV-GSBL, c = ',num2str(c)],'fontsize',12);

% Illustrate overall LS image
figure(8)
imshow(reshape(x_LS,N,N));
title('LS','fontsize',12)
colormap(gray)

% Illustrate overall IAS image
figure(9)
imshow(reshape(x_IAS,N,N));
%imshow(reshape(x_IAS,N,N), 'InitialMagnification',800);
title('IAS','fontsize',12)
colormap(gray)

% Illustrate overall MMV-IAS image
figure(10)
imshow(reshape(x_MMVIAS,N,N));
%imshow(reshape(x_MMVIAS,N,N), 'InitialMagnification',800);
title('MMV-IAS','fontsize',12)
colormap(gray)

% Illustrate overall GSBL image
figure(11)
imshow(reshape(x_GSBL,N,N));
%imshow(reshape(x_GSBL,N,N), 'InitialMagnification',800);
title('GSBL','fontsize',12)
colormap(gray)

% Illustrate overall MMV-GSBL image 
figure(12)
imshow(abs(reshape(x_MMVGSBL,N,N)));
%imshow(reshape(x_MMVGSBL,N,N), 'InitialMagnification',800);
title('MMV-GSBL','fontsize',12)
colormap(gray)