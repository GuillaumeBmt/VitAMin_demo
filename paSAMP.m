%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    phase aware Swept Approximate Message Passing (paSAMP) algorithm     %   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% authors : Guillaume Beaumont, Angélique Drémeau.                        %
% contact : guillaume.beaumont@ensta-bretagne.org                         %
%                                                                         %
% Last modification : 19/05/2020                                          %
% Last remark       : Cleaning code for sharing                           %
%                     Also deleted p entry careful while using main.m     %
%                                                                         %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT : 
%   Y   - measurement vector (matrician calculation not yet implemented)
%   H   - measurement matrix
%   opt - option structure  A COMPLETER
%          -> niter : max number of iteration
%          -> init_a : mean(X) vector at start
%          -> init_c : var(X) vector at start
%          -> var_n  : prior additive noise variance 
%          -> icov_theta : precision matrix of the phase noise
%          -> mean_p :  mean angular bias of phase noise
%          -> vnf : security value for too small quantities
%          -> rho : Bernoulli parameter
%          -> xm : mean value of the non-null coeff in X
%          -> xv : variance of the non-null coeff in X
% OUTPUT :                                                                %
%   a   - mean value of the distribution of posterior p(x:y)              %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [a] = paSAMP(Y,H,opt)

[M, N] = size(H);                    % Getting the dimmensionnal parameters

%% Get the initial values of posterior mean an variance
a = opt.init_a;
c = opt.init_c;

%% Initialization
g = 0;
S = zeros(N, 1);
R = zeros(N, 1);

absH2 = abs(H).^2;
conjH = conj(H);
%% MAIN LOOP
for i =1:opt.niter
    
    V =double(absH2*single(c));
    O = double(H*a - V.*g);
    
    [g, dg] = goutPa(Y,O,V,opt.delta,opt.meanremov,opt);
    
    g_old = g;
    a_oldies = a;
%% ITERATIVE LOOP AS PROPOSED BY MANOEL & KRZAKALA
    ind = randperm(N,N);
    for j=1:N
        k = ind(j);

        if(i>1)                              % Update with a damping factor
            
            S(k) = damping(1./(sum(absH2(:,k).*(-dg))), S(k), opt.damp);
            R(k) = damping(a(k)+S(k)*sum(conjH(:,k).* g), R(k), opt.damp);
        else                                 % First calculation for i = 1
            
            S(k) = 1./(sum(absH2(:,k).*(-dg)));
            R(k) = a(k)+S(k)*sum(conjH(:,k).* g);
        end

        a_old = a(k);
        c_old = c(k);


        [a(k), c(k)] = BernoulligaussianPrior(S(k),R(k),opt.rho,opt.xm,opt.xv,opt.vnf,a_old);
        if ~isfinite(a(k))
            a(k)=a_old;
            c(k)=c_old;
        end


        VOld = V;

        V = V + absH2(:,k)*(c(k)-c_old);

        O = O + H(:,k)*(a(k)-a_old) - g_old .* (V-VOld);


        [g, dg] = goutPa(Y,O,V,opt.delta,opt.meanremov,opt);
    end
%% END OF ITERATIVE LOOP

        
    
% EM estimation for multiplicative phase noise
    Z_est=H*a;
    Z_est_var=c'*diag(H'*H);
    
    if opt.adapttheta 
        zest = H*a;
        t    = conj(Y).*zest;
        
        icov_y = diag(sparse(2*abs(t)/opt.delta));        
        icov_p = icov_y+opt.icov_theta;
        
        var_p = diag(inv(icov_p));
        moy_p = icov_p\(-icov_y*atan2(imag(t),real(t))+opt.icov_theta*opt.mean_p);
        
% Update of the noise parameters 
        opt.mean_p  = moy_p;
        opt.varmarg = var_p;
    end
% Additive noise learning option   
    if opt.adaptdelta
        opt.delta = noiselearn(Y,Z_est,Z_est_var,opt,M);
    end   
       
%% CONVERGENCE CRITERION 
% Here the convergence criterion is chose as the correlation between two
% consecutive estimations:
    diff = 1-abs(a'*a_oldies)/(norm(a_oldies)*norm(a));
    
    if ~isfinite(diff)
        % If there is NaN in the result, keep the previous estimation
        disp('Inifinites appearing...');
        disp("converged at"+ i + " iterations " );
        a = a_oldies;
        break;
    end
    
    if(~isnan(diff)*diff < opt.converg)
        disp("converged at "+ i + " iterations " );
        break;
    end
    
    
    
    
    

    


end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CALCULATION OF THE OUTGOING MESSAGES
function [g, dg] = goutPa(Y,O,V,delta,meanremov,opt) 


if(meanremov)
    D =  delta;
    D(end-1:end) = 0;
else 
    D =  delta;
end


a=(V+D)./(2.*abs(Y).*abs(O));

thetam2=((-atan2(imag(conj(Y).*O),real(conj(Y).*O))./a)+(opt.mean_p./opt.varmarg))./((1./a)+(1./opt.varmarg));

SigmaT2=1./((1./a)+(1./opt.varmarg));


opt.mean_p= thetam2;
opt.varmarg= SigmaT2;

% Denoized messages z = Hx

E_zIy=((Y.*V)./(D+V)).*exp(-1i*thetam2).*(besseli(1,(1./SigmaT2))./besseli(0,(1./SigmaT2))) + ((O.*D)./(V+D));
var_zIy= (1./(abs(D + V))).*(abs(Y.*V).^2+abs(O.*D).^2+2.*(abs(Y.*V).*abs(O.*D).*(besseli(1,(1./SigmaT2))./besseli(0,(1./SigmaT2))).*cos(thetam2+atan2(imag(conj(Y).*O),real(conj(Y).*O)))))+((V.*D)./(V+D))-abs(E_zIy).^2;

g= (1./V).*(E_zIy-O);
dg=(1./V).*((var_zIy./V) -1);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DAMPING FUNCTION
function res = damping(newVal,oldVal,damp)
res=(1-damp).*newVal+damp.*oldVal;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CALCULATION OF THE INGOING MESSAGES
function [a, c] = BernoulligaussianPrior(S,R,rho,xm,xv,vnf,~)
% integration d'un prior Bernouilli gaussien au vecteur X.
% Calcul de la constante de normalisation du prior
Znor=rho.*sqrt(2*pi*((xv.*S)./abs(xv+S))).*exp((-abs(xm-R)^2)./(2.*abs(S+xv)))+(1-rho).*exp((-abs(R)^2)./abs(2.*S));
Znor(Znor<eps) = vnf;
% Calcul des moyennes et variances du nouveau prior
a=(1./Znor).*rho.*exp((-abs(xm-R)^2)./(2.*abs(S+xv))).*sqrt(2*pi*((xv.*S)./abs(xv+S))).*((xv.*R+S.*xm)./abs(S+xv));
c=(1./Znor).*rho.*exp((-abs(xm-R)^2)./(2.*abs(S+xv))).*sqrt(2*pi*((xv.*S)./abs(xv+S))).*(abs(((xv.*R+S.*xm)./abs(S+xv))).^2+((xv.*S)./abs(xv+S))) - abs(a).^2;
c(c<eps) = vnf;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ADDITIVE NOISE LEARNING
function [var_n] = noiselearn(Y,a,c,phase_opt,dim_y)
ybar= Y.*exp(-1i*phase_opt.mean_p).*R0(full(1./phase_opt.varmarg));
var_n=(1./(dim_y)).*(Y'*Y-2*real(ybar'*a)+a'*a+c);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function result = R0(phi)
num = besseli(1,phi,1);
 denom = besseli(0,phi,1);
result = num./denom;  
end

