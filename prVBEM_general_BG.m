function [x_hat,KLdiv]=prVBEM_general_BG(y,D,opt)
      

% Mean-Field approximation for phase retrieval with Gaussian prior model on
% the phases
%*********************************************
% This matlab function implements the phase recovery algorithm described in
% "XXX"
% by A. Dr�meau and C. Herzet
%
%
% AUTHORS: A. Dr�meau & C. Herzet
%
%
% INPUTS:
% y  : vector of observations
% D  : dictionary
% opt: options
% . var_a   : variance of the non-zero coefficients in x
% . var_n   : variance of Gaussian noise n (default: 10^-8))
% . pas_est : used in the stopping criterion to force the decrease of the
%            estimated noise variance (default: 0.1)
% . flag_est_n : if 'on', the noise variance is estimated at each
%                iteration (default:'on')
%
%
% OUTPUTS:
% x_hat: recovered signal
% KLdiv: value of the Kullback-Leibler divergence at stopping point
%




% Initialization
%*****************                       
[dim_y,dim_s]=size(D);

if nargin < 3, 
    opt_OK=0; 
    opt.x0         = D\y;
else 
    opt_OK=1;
end
if ~sum(strcmp(fields(opt),'var_a')) || ~opt_OK
    opt.var_a         = max(abs(D\y))^2;
    disp('var_a undefined: default value used')
end
if ~sum(strcmp(fields(opt),'var_n')) || ~opt_OK
    opt.var_n         = 10^-8;
    disp('var_n undefined: default value used')    
end
if ~sum(strcmp(fields(opt),'ps')) || ~opt_OK
    opt.ps            = 0.5*ones(dim_s,1);
    disp('var_n undefined: default value used')    
end
if ~sum(strcmp(fields(opt),'flag_est_n')) || ~opt_OK
    opt.flag_est_n    = 'on';
    disp('flag_est_n undefined: default value used')
end
if ~sum(strcmp(fields(opt),'flag_est_var_theta')) || ~opt_OK
    opt.flag_est_var_theta    = 'on';
    disp('flag_est_var_theta undefined: default value used')
end
if ~sum(strcmp(fields(opt),'niter')) || ~opt_OK
    opt.niter         = 600;
    disp('n_iter undefined: default value used')        
end
if ~sum(strcmp(fields(opt),'pas_est')) || ~opt_OK
    opt.pas_est       = 0.1;
    disp('pas_est undefined: default value used')
end
if ~sum(strcmp(fields(opt),'flag_cv')) || ~opt_OK
    opt.flag_cv       = 'KL';
    disp('flag_cv undefined: default value used')
end
if ~sum(strcmp(fields(opt),'icov_theta')) || ~opt_OK
    opt.icov_theta    = 10^(-9)*speye(dim_y);
    disp('icov_theta undefined: default value used')
end
if ~sum(strcmp(fields(opt),'moy_theta')) || ~opt_OK
    opt.moy_theta     = zeros(dim_y,1);
    disp('moy_theta undefined: default value used')
end



ps                  = opt.ps;
var_a               = opt.var_a;
var_n               = opt.var_n;
x0                  = opt.x0;
flag_est_n          = opt.flag_est_n;
flag_est_var_theta  = opt.flag_est_var_theta;
flag_cv             = opt.flag_cv;
pas_est             = opt.pas_est;
niter               = opt.niter;
icov_theta          = opt.icov_theta;
moy_theta           = opt.moy_theta;

if strcmp(flag_est_var_theta,'on')
    icov_theta_init = icov_theta;
end


if var_n==0
    var_n = 10^(-6);
    warning('Noise variance set to a small value for implementation reasons.')
end

qs      = zeros(dim_s,2);
moy_x   = zeros(dim_s,2);
var_x   = zeros(dim_s,2);
moy_x_q = zeros(dim_s,1);
var_x_q = zeros(dim_s,1);
w       = zeros(dim_s,1);
moy_p   = zeros(dim_y,1);
var_p   = zeros(dim_y,1);
z       = zeros(dim_y,1);
t       = zeros(dim_y,1);
ybar    = zeros(dim_y,1);
icov_y  = speye(dim_y);
icov_p  = speye(dim_y);


ybar(:)    = y;
w(:)       = D'*ybar;
H          = D'*D;
moy_x(:,1) = x0;
moy_x_q(:) = qs(:,1).*moy_x(:,1);
var_x(:,1) = var_a*var_n./(var_n+var_a*diag(H));
var_x(:,2) = var_a;
var_x_q(:) = (var_x(:,1)+abs(moy_x(:,1)).^2).*qs(:,1)-abs(moy_x_q(:)).^2;
z(:)       = D*moy_x_q;
var_n_true = var_n;


% Iterative process
%*******************
% Convergence criteria
OK_outer=1;
compt=1;
KLdiv_old=100000;

while OK_outer && compt<niter
        
    % Estimation of var_n  
    %*********************
    if strcmp(flag_est_n,'on') || strcmp(flag_est_n,'on_off') 
        var_n=inv(dim_y)*(z'*z+var_x'*diag(H)+y'*y-2*real(ybar'*z));
        if ~isreal(var_n)
            error('var_n is not real')
        end
    end
    
    
    % Update q(theta)=G(moy_p,icov_p^{-1}})
    %**************************************
    z(:) = D*moy_x_q;
    t(:) = conj(y).*z;
    
    icov_y(:) = diag(sparse(2*abs(t)/var_n));
    icov_p(:) = icov_y+icov_theta;
    var_p(:)  = diag(inv(icov_p));
    moy_p(:)  = icov_p\(-icov_y*angle(t)+icov_theta*moy_theta);%moy_p(1)

    I0 = besseli(zeros(dim_y,1),full(1./var_p));
    I1 = besseli(ones(dim_y,1),full(1./var_p));
    fac_bessel = I1./I0;
    if ~isempty(find(isnan(fac_bessel)==1,1)), fac_bessel(isnan(I1./I0)==1)=1; end    
    ybar(:)    = y.*exp(-1i*moy_p).*fac_bessel;
    w(:)       = D'*ybar;
    a_oldies = moy_x_q;
    
    % Update of the q(x_i,s_i)=G(moy_x,var_x)
    %****************************************
    for i=1:1:dim_s
        
        % Update of the q(x_i|s_i)=G(moy_x,var_x)
        val_tmp=w(i)-H(i,:)*moy_x_q+H(i,i)*moy_x_q(i);
        moy_x(i,1)=(var_a/(var_n+var_a*H(i,i)))*val_tmp;
        moy_x(i,2)=0;
        var_x(i,1)=var_n*var_a/(var_n+var_a*H(i,i));
        var_x(i,2)=var_a;
        
        % Update q(s_i)
        log_qs1_tmp=0.5*(abs(moy_x(i,1))^2)/var_x(i,1)+log(ps(i))+0.5*log(var_x(i,1));
        log_qs2_tmp=0.5*(abs(moy_x(i,2))^2)/var_x(i,2)+log(1-ps(i))+0.5*log(var_x(i,2));
        log_norm_fac=max(log_qs1_tmp,log_qs2_tmp)+log(1+exp(-abs(log_qs1_tmp-log_qs2_tmp)));
        qs(i,1)=exp(log_qs1_tmp-log_norm_fac);
        qs(i,2)=exp(log_qs2_tmp-log_norm_fac);
        
        moy_x_q(i)=moy_x(i,1)*qs(i,1);
        
    end
    
    var_x_q(:) = (var_x(:,1)+abs(moy_x(:,1)).^2).*qs(:,1)-abs(moy_x_q(:)).^2;
    z(:)       = D*moy_x_q;
       
    
    % Estimation of var_theta
    %************************
    if strcmp(flag_est_var_theta,'on')
        var_theta = trace(icov_theta_init*(inv(icov_p)+moy_p*moy_p'))/dim_y;
        if ~isreal(var_theta)
            error('var_theta is not real')
        end
        icov_theta(:) = icov_theta_init/var_theta;
    end
    % NMSE COnvergence Criterion
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    a = moy_x_q;
    % a_oldies voir ligne 172
    diff = 1-abs(a'*a_oldies)/(norm(a_oldies)*norm(a));
    %diff = norm(abs(a-a_oldies))/abs((norm(a)))
    
%     if ~isfinite(diff)
%         disp('Inifinites appearing...');
%         a = a_oldies;
%         OK_outer=0;
%     end
    
    if(~isnan(diff)*diff < opt.converg)
        disp("CV après "+ i + " ité " );
        OK_outer=0;
    end
    
    % Convergence checks
    %*******************            
%     if strcmp(flag_cv,'KL')
% 
%         % Computation of the KL divergence
%         % int log p(y,x) = int log p(y|x) + int log p(x)
%         frst = (dim_y)*log(var_n)+(1/var_n)*(z'*z+var_x_q'*diag(H)+y'*y-2*real(ybar'*z));
%         scnd = (dim_s)*log(var_a)+(1/var_a)*sum((var_x_q(:,1)+abs(moy_x_q(:,1)).^2));
%         
%         % int log q(x)
%         sxth = sum(log(var_x_q));
%         vec_tmp = abs(t).*fac_bessel;
%         I0_tmp=I0;
%         svth = (2/var_n)*sum(vec_tmp)-sum(log(nonzeros(I0_tmp)));
% 
%         KLdiv = frst + scnd - sxth + svth;
%         diff  = KLdiv_old - KLdiv;
%         
%         if isnan(diff) || isinf(abs(diff))
%             %warning('Stopping criterion changed to fixed number of iterations.')
%             flag_cv = 'iter';
%         end
% 
%         if diff<1e-8
%             if (var_n/var_n_true)>1.1
%                 var_n = var_n - pas_est*(var_n-var_n_true);
%                 flag_est_n = 'off';
% 
%                 % Computation of the KL divergence
%                 % int log p(y,x) = int log p(y|x) + int log p(x)
%                 frst = (dim_y)*log(var_n)+(1/var_n)*(z'*z+var_x_q'*diag(H)+y'*y-2*real(ybar'*z));
%                 scnd = (dim_s)*log(var_a)+(1/var_a)*sum((var_x_q(:,1)+abs(moy_x_q(:,1)).^2));
% 
%                 % int log q(x)
%                 sxth = sum(log(var_x_q));
%                 vec_tmp = abs(t).*fac_bessel;
%                 I0_tmp=I0;
%                 svth = (2/var_n)*sum(vec_tmp)-sum(log(nonzeros(I0_tmp)));
% 
%                 KLdiv = frst + scnd - sxth + svth;
% 
%             else
%                 OK_outer=0;
%             end              
%         end
%         
%         KLdiv_old = KLdiv;
%         
%     end
    compt=compt+1;
end
x_hat = moy_x_q;
KLdiv = KLdiv_old;


