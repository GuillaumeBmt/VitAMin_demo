%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Von Mises Swept Approximate Message Passing Algorithm             %   
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
% INPUT :                                                                 %
%   Y   - measurement vector (matrician calculation not yet implemented)  %
%   H   - measurement matrix                                              %
%   opt - option structure                                                %
%          -> niter : max number of iteration
%          -> init_a : mean(X) vector at start
%          -> init_c : var(X) vector at start
%          -> var_n  : prior additive noise variance 
%          -> icov_theta : precision matrix of the phase noise
%          -> bias_theta : kappa vector precision of phase noise
%          -> vnf : security value for too small quantities
%          -> rho : Bernoulli parameter
%          -> xm : mean value of the non-null coeff in X
%          -> xv : variance of the non-null coeff in X
% OUTPUT :                                                                %
%   a   - mean value of the distribution of posterior p(x:y)              %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [a] = VitAMin(Y,H,opt)

[dim_y, dim_s] = size(H);

%% Initialization

g = 0;
S = zeros(dim_s, 1);
R = zeros(dim_s, 1);
a = opt.init_a;
c = opt.init_c;
icov_p  = speye(dim_y);

%% MAIN LOOP

absH2 = abs(H).^2;
conjH = conj(H);

for i =1:opt.niter
    
    zest = H*a;
    t    = Y.*conj(zest);
    alpha_1 = sparse(2*t/opt.var_n);
    alpha_2 = zeros(dim_y,1);
    alpha_3 = zeros(dim_y,1);

    tmp_p = alpha_1 + (opt.bias_theta);
    moy_p = angle(tmp_p);
    abs_p = abs(tmp_p);

     for i=1:1:dim_y
        
        I0 = besseli(zeros(dim_y,1),abs_p);
        I1 = besseli(ones(dim_y,1),abs_p);
        fac_bessel = I1./I0;
        
        if ~isempty(find(isnan(fac_bessel)==1,1)), fac_bessel(isnan(I1./I0)==1) = 1; end   
        alpha_2(:) = opt.icov_theta*(sin(moy_p).*fac_bessel);
        alpha_3(:) = opt.icov_theta*(cos(moy_p).*fac_bessel);
        
        tmp_p(i) = alpha_1(i) + (opt.bias_theta(i));
        moy_p(i) = angle(tmp_p(i));
        abs_p(i) = abs(tmp_p(i));
        
      end

    ybar(:)    = Y.*exp(-1i*moy_p);

    var_theta = trace(opt.icov_theta*(inv(icov_p)+moy_p*moy_p'))/dim_y;
    
    if ~isreal(var_theta)
        error('var_theta is not real')
    end

    
    V =double(absH2*single(c));
    O = double(H*a - V.*g);
    
    [g, dg] = goutPa(Y,O,V,opt.var_n,opt.meanremov,opt.bias_theta,opt.icov_theta,opt.mean_p);
    

    g_old = g;
    a_oldies = a;
%% ITERATIVE LOOP AS PROPOSED BY MANOEL & KRZAKALA
    ind = randperm(dim_s,dim_s);
    for j=1:dim_s
        k = ind(j);

        if(i>1)
            S(k) = damping(1/(sum(absH2(:,k).*(-dg))), S(k), opt.damp);
            R(k) = damping(a(k)+S(k)*sum(conjH(:,k).* g), R(k), opt.damp);
        else
            S(k) = 1/(sum(absH2(:,k).*(-dg)));
            R(k) = a(k)+S(k)*sum(conjH(:,k).* g);
        end
        
        a_old = a(k);
        c_old = c(k);

        [a(k), c(k)] = BernoulligaussianPrior(S(k),R(k),opt.rho,opt.xm,opt.xv,opt.vnf,a_old);
        
        if ~isfinite(a(k))
            a(k)=a_old;
            c(k)=c_old;
        end

        if(opt.meanremov && k>N-2)
            a(k) = R(k);
            c(k) = S(k);
        end

        VOld = V;

        V = V + absH2(:,k)*(c(k)-c_old);

        O = O + H(:,k)*(a(k)-a_old) - g_old .* (V-VOld);


        [g, dg] = goutPa(Y,O,V,opt.var_n,opt.meanremov,opt.bias_theta,opt.icov_theta,opt.mean_p);
    end

    %% CONVERGENCE CRITERION : NORMALIZED CORRELATION ERRROR BETWEEN TWO SUBSEQUENT ESTIMATIONS
    
    diff = 1-abs(a'*a_oldies)/(norm(a_oldies)*norm(a));
    
    if ~isfinite(diff)
        disp('Inifinites appearing...');
        disp("converged at "+ i + " iterations " );
        a = a_oldies;
        break;
    end
    
    if(~isnan(diff)*diff < opt.converg)
        disp("converged at "+ i + " iterations " );
        break;
    end
    
    
    
    Z_est=H*a;
    Z_est_var=c'*diag(H'*H);
    %% EM ESTIMATION OF ADDITIVE NOISE
    if opt.adaptdelta
        opt.var_n = noiselearn(Y,Z_est,Z_est_var,opt,dim_y,ybar);
    end   
    
end
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CALCULATION OF THE OUTGOING MESSAGES
function [g, dg] = goutPa(Y,O,V,var_n,meanremov,bias_theta,icov_theta,mean_p) % CHECKED

% bias_theta ( vecteur taille n )
% icov_theta ( matrice de cov nxn) : rappel , diagonale nulle

format long 




if(meanremov)
    D =  var_n;
    D(end-1:end) = 0;
else 
    D =  var_n;
end


alpha1=(abs(Y).*abs(O))./(V+D);

Abob = abs(alpha1.*exp(-1i.*atan2(imag(conj(Y).*O),real(conj(Y).*O)))+(bias_theta).*exp(1i.*mean_p));
%thetabob = angle(alpha1.*exp(-1i.*atan2(imag(conj(Y).*O),real(conj(Y).*O)))+(bias_theta-alpha3).*exp(1i.*mean_p));
thetabob = atan2(imag(alpha1.*exp(-1i.*atan2(imag(conj(Y).*O),real(conj(Y).*O)))+(bias_theta).*exp(1i.*mean_p)),real(alpha1.*exp(-1i.*angle(conj(Y).*O))+(bias_theta).*exp(1i.*mean_p)));


E_zIy=((Y.*V)./(D+V)).*exp(-1i*thetabob).*(R0(Abob)) + ((O.*D)./(V+D));

%var_zIy=(((abs(Y.*V.*exp(-1i*thetam2)+O.*D))./(abs(D + V))).^2).*R0(1./SigmaT2)+((V.*D)./(V+D))-abs(E_zIy).^2;
var_zIy= (1./(abs(D + V).^2)).*(abs(Y.*V).^2+abs(O.*D).^2+2.*(abs(Y.*V).*abs(O.*D).*(R0(Abob)).*cos(thetabob+atan2(imag(conj(Y).*O),real(conj(Y).*O)))))+((V.*D)./(V+D))-abs(E_zIy).^2;
% var_zIy(var_zIy<0)=vnf;

g= (1./V).*(E_zIy-O);
dg=(1./V).*((var_zIy./V) -1);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function res = damping(newVal,oldVal,damp)
res=(1-damp).*newVal+damp.*oldVal;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DENOISING FUNCTION / CALCULATION OF THE INGOING MESSAGES

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

function [var_n] = noiselearn(Y,a,c,opt,dim_y,y_bar)
%var_n=(1./(dim_y)).*(a'*a+c+Y'*Y-2*real(y_bar*a));
var_n=(1./(dim_y)).*(a'*a+Y'*Y-2*real(y_bar*a));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function result = R0(phi)
num = besseli(1,phi,1);
 denom = besseli(0,phi,1);
result = num./denom;  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


