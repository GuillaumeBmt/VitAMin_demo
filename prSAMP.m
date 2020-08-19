function [a] = prSAMP(Y,H,opt)

[M, N] = size(H);

%% Get parameters
rho = opt.rho;
delta = opt.delta;
display = opt.display;
converg = opt.converg;
damp = opt.damp;
amptype = opt.amptype;
a = opt.init_a;
c = opt.init_c;
vnf = opt.vnf;
niter = opt.niter;
meanremov = opt.meanremov;
adaptdelta = opt.adaptdelta;

%% Initialization
g = 0;
S = zeros(N, 1);
R = zeros(N, 1);

%% main loop
absH2 = abs(H).^2;
conjH = conj(H);

for i =1:niter
    
    V =absH2*single(c);
    O = H*a - V.*g;
    
    [g, dg] = gout2(Y,O,V,delta,vnf,meanremov);
    
    if(strcmp(amptype, 'SwAMP'))
        g_old = g;
        aoldies = a;
        ind = randperm(N,N);
        
        for j=1:N
            k = ind(j);
            
            if(i>1)
                S(k) = damping2(1/sum(absH2(:,k).*dg), S(k), damp);
            else
                S(k) = 1/sum(absH2(:,k).*dg);
            end
            
            if S(k)<0
                S(k) = 0.1*vnf;
            end
            
            if(i>1)
                R(k) = damping2(a(k)+S(k)*sum(conjH(:,k).* g), R(k), damp);
            else
                R(k) = a(k)+S(k)*sum(conjH(:,k).* g);
            end
            
            a_old = a(k);
            c_old = c(k);
            
            [a(k), c(k)] = BernoulligaussianPrior(S(k),R(k),rho,opt.xm,opt.xv,vnf,a_old);
            
            if(meanremov && k>N-2)
                a(k) = R(k);
                c(k) = S(k);
            end
            
            VOld = V;
            V = V + absH2(:,k)*(c(k)-c_old);
            O = O + H(:,k)*(a(k)-a_old) - g_old .* (V-VOld);
            
            [g, dg] = gout2(Y,O,V,delta,vnf,meanremov);
            
        end
        
    else
        if(i>1)
            S = damping2(squeeze(sum(bsxfun(@times,absH2,permute(dg,[1,3,2])),1)).^(-1), S', damp);
        else
            S = squeeze(sum(bsxfun(@times,absH2,permute(dg,[1,3,2])),1)).^(-1);
        end
        
        S(S<0) = 0.1*vnf;
                
        S = S.';
        if(i>1)
            R = damping2(a+S.*sum(bsxfun(@times,conjH,g),1).', R, damp);
        else
            R = a+S.*sum(bsxfun(@times,conjH,g),1).';
        end
        
        a_old = a;
        
        [a, c] = BernoulligaussianPrior(S(k),R(k),rho,opt.xm,opt.xv,vnf,a_old);
        
        if(meanremov)
            a(end-1:end) = R(end-1:end);
            c(end-1:end) = S(end-1:end);
        end
        
    end
    
    
    %% Check stoping conditions
    %diff = norm(abs(a-a_old))/abs((norm(a)));
    diff = abs(a'*aoldies/(norm(aoldies)*norm(a)));
    
    if ~isfinite(diff)
        disp('Inifinites appearing...');
	a = aoldies;
        break;
    end
    
    if(~isnan(diff)*diff > converg)
        disp("CV après "+ i + " ité " )
        break;
        
    end
    
    
    %% difference between y and |G*x_est|
    Y_est = abs(H*a);
    err = 1/M*sum((Y - Y_est).^2)/var(Y);
    
    if adaptdelta
        delta = var(Y - Y_est);
    end
    
    %% Display what is happening
    if  display
        if isfield(opt,'signal')
            mse_x = sum((a-opt.signal).^2)/N;
            fprintf('Iteration %d , difference %d, MSE_x: %d, MSE_y: %d\n',i,diff,mse_x,err);
        else
            fprintf('Iteration %d , difference %d, MSE_y: %d\n',i,diff,err);
        end
    end
    
    
end

function res = damping2(newVal,oldVal,damp)

res=(1-damp).*newVal+damp.*oldVal;


function [g, dg] = gout2(Y,O,V,delta,vnf,meanremov)
O=max(O,eps);

if(meanremov)
    D = ones(length(V), 1) * delta;
    D(end-1:end) = 0;
else 
    D = ones(length(V), 1) * delta;
end

phi = 2*abs(O).*abs(Y)./(D+V);
g=O./(D+V).*(abs(Y)./abs(O).*R0(phi) - 1); 

var = abs(Y).^2.*(1-R0(phi).^2)./(1+D./V).^2 + D.*V./(D+V);

var(var<eps)=0.1*vnf;
dg =  1./V.*(var./V -1);

function [a, c] = BernoulligaussianPrior(S,R,rho,xm,xv,vnf,a_old)
% int�gration d'un prior Bernouilli gaussien au vecteur X. 
SIGMA2=double((xv.*S)./abs(xv+S));
MOYENNE=double((xv.*R+S.*xm)./abs(S+xv));
% logVAR=double(log(1-rho)+((-abs(R).^2)./(2.*S))-log(rho.*(2.*(S+xv).*sqrt(2*pi*SIGMA2)))-(-(abs(xm-R).^2)));
% loga=double(log(MOYENNE)-log(1+exp(logVAR)));
% logc=log(MOYENNE.^2+SIGMA2)-log(1+exp(logVAR))+log(1+((MOYENNE.^2./(MOYENNE.^2+SIGMA2))./(1+exp(logVAR))));

Znor=rho.*exp((-abs(xm-R)^2)./(2.*abs(S+xv))).*sqrt(2*pi*SIGMA2)+(1-rho).*exp((-abs(R)^2)./abs(2.*S));
Znor(Znor<eps) = vnf;
a=(1./Znor).*rho.*exp((-abs(xm-R)^2)./(2.*abs(S+xv))).*sqrt(2*pi*SIGMA2).*MOYENNE;

c=(1./Znor).*rho.*exp((-abs(xm-R)^2)./(2.*abs(S+xv))).*sqrt(2*pi*SIGMA2).*(abs(MOYENNE).^2+SIGMA2) - abs(a).^2;


c(c<eps) = vnf;
c(~isfinite(c))= vnf;
% a=exp(loga);
% c=exp(logc);
if ~isfinite(a)
%     disp('Nan detected in a');

    
    
end
if ~isfinite(c)
%     disp('Nan detected in c');

    
    
end

function result = R0(phi)


num = besseli(1,phi,1);
denom = besseli(0,phi,1);
result = num./denom;









