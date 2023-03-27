%% MMV_GSBL_1d
%
% Description: 
%  Function for the MMV-GSBL algorithm. 
%  We assume a one-dimensional signal and iid noise
%
% INPUT: 
%  J :          number of MMVs
%  F :          list of forward operator 
%  y :          MMVs 
%  variance :   list of noise variances 
%  R :          regularization matrix
%  c, d :       regualrization hyper-hyper-parameters 
%
% OUTPUT: 
%  x :          list of MAP estimates for x 
%  beta :     	MAP estimate for common hyper-parameter vector beta 
%  history :    structure containing residual norms and the tolerances at each iteration
%
% Author: Jan Glaubitz 
% Date: Jan 17, 2023 
% 

function [x, beta, history] = MMV_GSBL_1d( J, F, y, variance, R, c, d, QUIET )

    t_start = tic; % measure time 

    %% Global constants and defaults  
    %MIN_ITER = 10; 
    MAX_ITER = 1000; 
    TOL_x = 1e-6;
    TOL_G = 1e-6;
    
    %% Data preprocessing & initial values 
    N = size(R,2); % number of pixels 
    K = size(R,1); % number of outputs of the regularization operator 
    beta = ones(K,1); % initial value for common hyper-parameter vector 
    beta_OLD = ones(K,1); % auxilary variable to compute change in beta 
    parfor j=1:J 
        FtF{j} = F{j}'*F{j}; % product corresponding to the forward operator 
        Fty{j} = F{j}'*y{j}; % forward operator applied to the indirect data 
        M{j} = size(F{j},1); % number of (indirect) measurements
        alpha{j} = 1/variance{j}; 
        x_OLD{j} = zeros(N,1); % mean 
        G{j} = @(x,beta) (1/variance{j})*norm( F{j}*x - y{j} ).^2/2 + norm( diag(beta.^(1/2))*R*x ).^2/2 + ... 
        sum( beta.*d ) - sum( ( c - 1/2 ).*log(beta) ); % objective function
    end 
   
    %% Outputting the learning progress 
    if ~QUIET
        fprintf('%3s\t%10s\t%10s\t%10s\t%10s\n', ... 
            'iter', 'change in x', 'tol x', 'change in G', 'tol G');
    end
    
    %% Iterate between the update steps until convergence of max number of iterations 
    for counter = 1:MAX_ITER

        % 1) x-updates 
        B = sparse(diag(beta)); % precision matrix 
        parfor j=1:J
            Gamma_inv{j} = sparse(real( alpha{j}*FtF{j} + R'*B*R )); % coefficient matrix
            x{j} = Gamma_inv{j}\( alpha{j}*Fty{j} ); % update x 
        end

        % 2) alpha-update 
        %parfor j=1:J
        %    alpha{j} = ( M{j}/2 + c - 1 )./( norm(F{j}*mu{j}-y{j})^2/2 + d );  
        %end

        % 3) beta-update 
        r = 0;
        for j=1:J 
            r = r + (real(R*x{j})).^2; 
        end
        beta = ( J/2 + c - 1 )./( r/2 + d ); 

        % store certain values in history structure 
        history.change_x(counter) = 0; 
        history.change_G(counter) = 0; 
        for j=1:J 
            aux_x = norm( x{j} - x_OLD{j} )/norm( x_OLD{j} ); % relative change in x{j}
            history.change_x(counter) = history.change_x(counter) + aux_x; % sum of relative changes
            aux_G = norm( G{j}(x{j},beta) - G{j}(x_OLD{j},beta_OLD) )/norm( G{j}(x_OLD{j},beta_OLD) ); % relative change in G
            history.change_G(counter) = history.change_G(counter) + aux_G; % sum of relative changes
            x_OLD{j} = x{j}; % store value of mu 
        end

        % display these values if desired 
        if ~QUIET
            fprintf('%3d\t%0.2e\t%0.2e\t%0.2e\t%0.2e\n', ... 
                counter, history.change_x(counter), TOL_x, ... 
                history.change_G(counter), TOL_G);
        end
        
        % check for convergence 
        if ( history.change_x(counter) < TOL_x || ...
                history.change_G(counter) < TOL_G )%&& ... 
                %counter > MIN_ITER )
             break;
        end
        
    end

    % output the time it took to perform all operations 
    if ~QUIET
        toc(t_start);
    end
    
end