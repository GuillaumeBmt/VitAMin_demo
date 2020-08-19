 function demo_code

close all
clear 
clc

for varp = [10^-2 10^-1 1]
%addpath(genpath('./CircStat2012a/'))
dim_y      = 32;
dim_s      = 64; %50;

% param simu 
param_name = 'noise_variance';
param_vec  = 10.^(-3:0.5:1)';
%param_vec  = 10.^[-5:0.5:-2]';
%param_vec  = [0:0.1:0.6]';
n_param    = length(param_vec);
n_algo     = 7;
n_sources = [1 2 5] ;

Perf_CorrNorm = zeros(n_algo,length(param_vec),length(n_sources));
Perf_JCC       = zeros(n_algo,length(param_vec),length(n_sources));

PerfSq_CorrNorm = zeros(n_algo,length(param_vec),length(n_sources));
PerfSq_JCC       = zeros(n_algo,length(param_vec),length(n_sources));

% Var_CorrNorm = zeros(n_algo,length(param_vec),length(n_sources));
% Var_JCC       = zeros(n_algo,length(param_vec),length(n_sources));

for posk          = [1 2 3]       % number of nonzero coeffcient in s
k = n_sources(posk);
disp(k)

n_trial    = 200;
var_x      = 1;       % variance coeff non nuls complexe
mean_x     = 0 + 1i.*0;                 % mean value of the non-zero coefficients 
% var_n      = 1e-3;    % variance bruit additif complexe
wavelength = 4;       % longueur d'onde normalisee (> 2)
flag_dico  = 'DoA';   % 'RandMod1', 'DoA'
flag_covp  = '1DVM';  % 'iid','Markov','MarkovRand','full','complexNoise', '1DVM', 'MVM'


% dico 
dico_opt.wavelength = wavelength;
dico_opt.flag_dico  = flag_dico;


% param algo
algo_opt.var_a      = var_x;
algo_opt.flag_est_n = 'off';
algo_opt.niter      = 800;
algo_opt.pas_est    = 0.1;
algo_opt.flag_cv    = []; %'KL'
algo_opt.converg= 10^-15;


% init
D     = dicoGen(dim_y,dim_s,dico_opt);
s     = zeros(dim_s,1);         % support
x     = zeros(dim_s,1);         % nonzero coef
x_hat1 = zeros(dim_s,1);            % estimates of x (one per algorithm)
x_hat2 = zeros(dim_s,1);
x_hat3 = zeros(dim_s,1); 
x_hat4 = zeros(dim_s,1); 
x_hat5 = zeros(dim_s,1); 
x_hat6 = zeros(dim_s,1);            % estimates of x (one per algorithm)
x_hat7 = zeros(dim_s,1);



y     = zeros(dim_y,1);         % obs
n     = zeros(dim_y,1);         % noise
p     = zeros(dim_y,1);         % phases





for cpt_param=1:n_param
    disp(fprintf([param_name ' = %3.4f' param_vec(cpt_param)]))
    disp(' ')
    algo_opt.var_n      =param_vec(cpt_param);
    var_n = algo_opt.var_n;
    
    for cpt_trial = 1:n_trial
        s(:) = 0;
        x(:) = 0;
        
        idx_perm = randperm(dim_s);
        idx_nz   = idx_perm(1:k);
        
        s(idx_nz)    = 1;
        x(idx_nz)    = mean_x + sqrt(var_x)*randn(k,1)+1i*sqrt(var_x)*randn(k,1);
        n(:)         = sqrt(0.5*var_n)*randn(dim_y,1)+1i*sqrt(0.5*var_n)*randn(dim_y,1);
        if strcmp(flag_covp,'complexNoise')
            dz = 0.2; % normalized distance between 2 sensors
            coherence_length = RCOH; % normalized coherence length (i.e. coherence length divided by the wavelength)
            f = @(x,y) exp(-0.5* ((x-y)*dz/coherence_length).^2);
            cov_phase = zeros(dim_y);
            for n1 = 1:dim_y
                for n2 = n1:dim_y
                    cov_phase(n1,n2) = f(n1,n2);
                    cov_phase(n2,n1) = cov_phase(n1,n2);
                end
            end 
            cov_phase(:,:) = cov_phase + 1e-12*eye(dim_y);
            
            m_phase    = zeros(dim_y,1);
            icov_phase = inv(cov_phase); % precision matrix for gaussian-like algo
            
            
            p(:)       = genPhaseRandom(dim_y,m_phase,cov_phase, flag_covp);
            y(:)       = exp(1i*p).*(D*x)+n;
            
%         elseif strcmp(flag_covp, 'MVM')
%             dz = 0.2; % normalized distance between 2 sensors
%             coherence_length = 5; % normalized coherence length (i.e. coherence length divided by the wavelength)
%             nsamples   = 50; 
%             
%             f = @(x,y) exp(-0.5* ((x-y)*dz/coherence_length).^2);
%             cov_phase = zeros(dim_y);
%             for n1 = 1:dim_y
%                 for n2 = n1:dim_y
%                     cov_phase(n1,n2) = f(n1,n2);
%                     cov_phase(n2,n1) = cov_phase(n1,n2);
%                 end
%             end 
%             cov_phase(:,:) = cov_phase + 1e-3*eye(dim_y);
%             
%             m_phase    = zeros(dim_y,1);
%             icov_phase = inv(cov_phase); % precision matrix for gaussian-like algo
%             
%             if n_trial<=nsamples
% %                 file_save = ['MVM_phase_dimObs_' num2str(dim_y) '_dz_' num2str(dz) '_coherenceLength_' num2str(coherence_length) '_nbtrials_' num2str(nsamples) '.mat'];
%                 file_save = ['MVM_phase_dimObs_' num2str(dim_y) '_dz_' num2str(dz) '_coherenceLength_' num2str(coherence_length) '_nbtrials_' num2str(nsamples) '_invMatrix.mat'];
%             else
%                 error('Error: not enough trials for MVM generated phases')
%             end
%             
%             load(['./Gibbs_MVM/' file_save], 'p')
%             
%             y(:) = exp(1i*p(:,cpt_trial)).*(D*x)+n;
%            
            
        else % gaussian model
            var_p1     = 10^-1; %10^6;    % variance of the initial phase (Markov)
            var_pt     = varp; %10^-0;   % variance of transition (Markov)
            drift      = 1;     % multiplicative factor (Markov)
            %aram phase model
            phase_opt.var_p1     = var_p1;
            phase_opt.var_pt     = var_pt;
            phase_opt.drift      = drift;
            phase_opt.flag_covp  = flag_covp;            
              
            [m_phase,icov_phase] = paramPhaseRandom(dim_y, phase_opt);
            p(:)         = genPhaseRandom(dim_y,m_phase,icov_phase, flag_covp);
            y(:)         = exp(1i*p).*(D*x)+n;
        end
        
        initbayes = eps*ones(dim_s,1)+eps*1i*ones(dim_s,1);
        
        % LS -> conventional BF
        %%%%%%%%%%%%%%%%%%%%%%%
        x_hat1(:)           = D\y;
        fprintf('pseudo-inverse     = %1.2f\n',abs(x_hat1'*x)/(norm(x_hat1)*norm(x)))
        
        Perf_CorrNorm(1,cpt_param,posk)=Perf_CorrNorm(1,cpt_param,posk) + abs(x_hat1'*x/(norm(x_hat1)*norm(x))); % NormCorr ( metrique de performances de reconstruction de vecteur       
        Perf_JCC(1,cpt_param,posk)      =Perf_JCC(1,cpt_param,posk) + JCC(abs(x),abs(x_hat1),posk);

        PerfSq_CorrNorm(1,cpt_param,posk)=PerfSq_CorrNorm(1,cpt_param,posk) + abs(x_hat1'*x/(norm(x_hat1)*norm(x)))*abs(x_hat1'*x/(norm(x_hat1)*norm(x))); % NormCorr ( metrique de performances de reconstruction de vecteur       
        PerfSq_JCC(1,cpt_param,posk)      =PerfSq_JCC(1,cpt_param,posk) + JCC(abs(x),abs(x_hat1),posk)*JCC(abs(x),abs(x_hat1),posk);

            
        %x_hat_LS           = x_hat1;

        
        % prVBEM (Gaussian prior on sources, uniform prior on phases)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        algo_opt.x0         = initbayes;
        algo_opt.moy_theta  = [];
        algo_opt.icov_theta = [];        
        [x_hat2(:),~]        = prVBEM(y,D,algo_opt);
        fprintf('VBEM uniform       = %1.2f\n',abs(x_hat2'*x)/(norm(x_hat2)*norm(x)))
        Perf_CorrNorm(2,cpt_param,posk)   = Perf_CorrNorm(2,cpt_param,posk)+abs(x_hat2'*x/(norm(x_hat2)*norm(x)));
        Perf_JCC(2,cpt_param,posk)      =Perf_JCC(2,cpt_param,posk) + JCC(abs(x),abs(x_hat2),posk);


        PerfSq_CorrNorm(2,cpt_param,posk)=PerfSq_CorrNorm(2,cpt_param,posk) + abs(x_hat2'*x/(norm(x_hat2)*norm(x)))*abs(x_hat2'*x/(norm(x_hat2)*norm(x))); % NormCorr ( metrique de performances de reconstruction de vecteur       
        PerfSq_JCC(2,cpt_param,posk)      =PerfSq_JCC(2,cpt_param,posk) + JCC(abs(x),abs(x_hat2),posk)*JCC(abs(x),abs(x_hat2),posk);


        % relaxed paVBEM (Gaussian prior on sources, Markov prior on phases)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        algo_opt.x0         = initbayes;
        algo_opt.moy_theta  = m_phase;
        algo_opt.icov_theta = icov_phase;
        algo_opt.flag_est_var_theta    = 'on';
        [x_hat3(:),~]        = prVBEM_general(y,D,algo_opt);
        fprintf('VBEM Markov        = %1.2f\n',abs(x_hat3'*x)/(norm(x_hat3)*norm(x)))                
        Perf_CorrNorm(3,cpt_param,posk)  = Perf_CorrNorm(3,cpt_param,posk)+abs(x_hat3'*x/(norm(x_hat3)*norm(x)));
        Perf_JCC(3,cpt_param,posk)      =Perf_JCC(3,cpt_param,posk) + JCC(abs(x),abs(x_hat3),posk);


        PerfSq_CorrNorm(3,cpt_param,posk)=PerfSq_CorrNorm(3,cpt_param,posk) + abs(x_hat3'*x/(norm(x_hat3)*norm(x)))*abs(x_hat3'*x/(norm(x_hat3)*norm(x))); % NormCorr ( metrique de performances de reconstruction de vecteur       
        PerfSq_JCC(3,cpt_param,posk)      =PerfSq_JCC(3,cpt_param,posk) + JCC(abs(x),abs(x_hat3),posk)*JCC(abs(x),abs(x_hat3),posk);


        % paVBEM (BG prior on sources, Markov model on phases)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        algo_opt.x0         = initbayes;
        algo_opt.ps         = k/dim_s*ones(dim_s,1);
        algo_opt.moy_theta  = m_phase;
        algo_opt.icov_theta = icov_phase;
        algo_opt.flag_est_var_theta    = 'on';
        [x_hat4(:),~]        = prVBEM_general_BG(y,D,algo_opt);
        fprintf('VBEM Markov BG     = %1.2f\n',abs(x_hat4'*x)/(norm(x_hat4)*norm(x)))
        Perf_CorrNorm(4,cpt_param,posk)  = Perf_CorrNorm(4,cpt_param,posk)+abs(x_hat4'*x/(norm(x_hat4)*norm(x)));
        Perf_JCC(4,cpt_param,posk)       = Perf_JCC(4,cpt_param,posk) + JCC(abs(x),abs(x_hat4),posk);


        PerfSq_CorrNorm(4,cpt_param,posk)=PerfSq_CorrNorm(4,cpt_param,posk) + abs(x_hat4'*x/(norm(x_hat4)*norm(x)))*abs(x_hat4'*x/(norm(x_hat4)*norm(x))); % NormCorr ( metrique de performances de reconstruction de vecteur       
        PerfSq_JCC(4,cpt_param,posk)      =PerfSq_JCC(4,cpt_param,posk) + JCC(abs(x),abs(x_hat4),posk)*JCC(abs(x),abs(x_hat4),posk);


%         x_hat_Markov = x_hat;



        icov_theta = icov_phase;
        algo_opt.icov_theta = []; 
        % VITAMIN ALGORITHM
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        algo_opt.xm        = mean_x;
        algo_opt.xv        = var_x;
        algo_opt.damp      = 0.75;                                      % Facteur de Damping ( optimal à 0.9 / va savoir ) 
        algo_opt.learning  = 0;                                        % Apprentisage de QUELQUE CHOSE
        algo_opt.rho       = k/dim_s;                               % Taux de parcimonie
        algo_opt.delta     = var_n;                                    % Variance du Bruit Additif
        algo_opt.display   = 0;                                        % Affichage de l'évolution de l'Algo
        algo_opt.niter     = algo_opt.niter;                           % Nombre d'itération du Message passing
        algo_opt.amptype   = 'SwAMP';                                  % Type d'algo / 'SwAMP'
        algo_opt.init_a    = initbayes; %                      % Initialisation de la moyenne de Xhat
        algo_opt.init_c    = 100*ones(dim_s,1);                        % Initialisation de la variance de Xhat
        algo_opt.vnf       = eps;                                 % Facteur de correction de variance 
        algo_opt.meanremov = 0;
        algo_opt.adapttheta= 0;                                        
        algo_opt.adaptdelta = 1;                                       % Adaptative step on additive noise ON
        algo_opt.signal     = x;
        algo_opt.icov_theta = icov_phase - diag(diag(icov_phase)); % Lambda_{mn} = Gamma^{-1}_{mn} if m=/=n and Lambda_{mm} = 0
        algo_opt.bias_theta = 2*(diag(icov_phase));
        algo_opt.mean_p     = m_phase;

        % Execution de l'algorithme 

        x_hat5(:) = VitAMin(y,D,algo_opt);
        fprintf('VITAMIN        = %1.2f\n',abs(x_hat5'*x)/(norm(x_hat5)*norm(x)));
        %perfxAddxPhase(4,cpt_param,k)  = perfxAddxPhase(4,cpt_param,k)+abs(x_hat4'*x/(norm(x_hat4)*norm(x)));
            Perf_CorrNorm(5,cpt_param,posk)  = Perf_CorrNorm(5,cpt_param,posk)+abs(x_hat5'*x/(norm(x_hat5)*norm(x)));
            Perf_JCC(5,cpt_param,posk)      =Perf_JCC(5,cpt_param,posk) + JCC(abs(x),abs(x_hat5),posk);
            
            
            PerfSq_CorrNorm(5,cpt_param,posk)=PerfSq_CorrNorm(5,cpt_param,posk) + abs(x_hat5'*x/(norm(x_hat5)*norm(x)))*abs(x_hat5'*x/(norm(x_hat5)*norm(x))); % NormCorr ( metrique de performances de reconstruction de vecteur       
            PerfSq_JCC(5,cpt_param,posk)      =PerfSq_JCC(5,cpt_param,posk) + JCC(abs(x),abs(x_hat5),posk)*JCC(abs(x),abs(x_hat5),posk);

            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        algo_opt.icov_theta = []; 
       % PR SAMP 
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        opt_pSAMP.xm        = mean_x;
        opt_pSAMP.xv        = var_x;
        opt_pSAMP.damp      = 0.75;                                     % Facteur de Damping ( optimal a 0.9 -empirical experiments-) 
        opt_pSAMP.learning  = 0;                                       % Relicat du PrSAMP
        opt_pSAMP.rho       = k/dim_s;                                 % Taux de parcimonie
        opt_pSAMP.delta     = var_n;                                   % Variance du Bruit Additif
        opt_pSAMP.display   = 0;                                       % Affichage de l'�volution de l'Algo
        opt_pSAMP.converg   = algo_opt.converg;                                    % Seuil de convergence de l'algorithme
        opt_pSAMP.niter     = algo_opt.niter;                          % Nombre d'it�ration du Message passing
        opt_pSAMP.amptype   = 'SwAMP';                                 % Type d'algo / 'SwAMP'
        opt_pSAMP.init_a    = initbayes;                               % Initialisation de la moyenne de Xhat
        opt_pSAMP.init_c    = 100*ones(dim_s,1);                       % Initialisation de la variance de Xhat
        opt_pSAMP.vnf       = 0.5;                                     % Facteur de correction de variance 
        opt_pSAMP.meanremov = 0;                                       % Mean Removal ( not used here )
        opt_pSAMP.adapttheta= 1;                                       % EM step for phase structure Estimation
        opt_pSAMP.adaptdelta = 1;                                      % Adaptative step
        opt_pSAMP.signal     = x;
            
        % Execution de l'algorithme

        x_hat6(:) = prSAMP(abs(y),D,opt_pSAMP);
        fprintf('prSAMP        =%1.2f\n',abs(x_hat6'*x)/(norm(x_hat6)*norm(x)));
        
        Perf_CorrNorm(6,cpt_param,posk)  = Perf_CorrNorm(6,cpt_param,posk)+abs(x_hat6'*x/(norm(x_hat6)*norm(x)));
        Perf_JCC(6,cpt_param,posk)      =Perf_JCC(6,cpt_param,posk) + JCC(abs(x),abs(x_hat6),posk);


        PerfSq_CorrNorm(6,cpt_param,posk)=PerfSq_CorrNorm(6,cpt_param,posk) + abs(x_hat6'*x/(norm(x_hat6)*norm(x)))*abs(x_hat6'*x/(norm(x_hat6)*norm(x))); % NormCorr ( metrique de performances de reconstruction de vecteur       
        PerfSq_JCC(6,cpt_param,posk)      =PerfSq_JCC(6,cpt_param,posk) + JCC(abs(x),abs(x_hat6),posk)*JCC(abs(x),abs(x_hat6),posk);


        
        
        %PASAMP
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %paSAMP with EM estimation of phase prior
        opt_SAMP.xm        = mean_x;
        opt_SAMP.xv        = var_x;
        opt_SAMP.damp      = 0.75;                                      % Facteur de Damping ( optimal à 0.9 / va savoir ) 
        opt_SAMP.learning  = 0;                                        % Apprentisage de QUELQUE CHOSE
        opt_SAMP.rho       = k/dim_s;                               % Taux de parcimonie
        opt_SAMP.delta     = var_n;                                    % Variance du Bruit Additif
        opt_SAMP.display   = 0;                                        % Affichage de l'évolution de l'Algo
        opt_SAMP.converg   = algo_opt.converg;                                     % Seuil de convergence de l'algorithme
        opt_SAMP.niter     = algo_opt.niter;                           % Nombre d'itération du Message passing
        opt_SAMP.amptype   = 'SwAMP';                                  % Type d'algo / 'SwAMP'
        opt_SAMP.init_a    = initbayes; %                      % Initialisation de la moyenne de Xhat
        opt_SAMP.init_c    = 100*ones(dim_s,1);                        % Initialisation de la variance de Xhat
        opt_SAMP.vnf       = eps;                                 % Facteur de correction de variance 
        opt_SAMP.meanremov = 0;
        opt_SAMP.adapttheta= 0;                                        
        opt_SAMP.adaptdelta = 1;                                       % Adaptative step on additive noise ON
        opt_SAMP.signal     = x;
        opt_SAMP.moy_theta   = 0;
        opt_SAMP.icov_theta  = icov_theta;
        opt_SAMP.varmarg     = diag(inv(icov_theta));
        opt_SAMP.mean_p=       m_phase;
        opt_SAMP.adapttheta= 1;                                        
        opt_SAMP.adaptdelta = 1;                                       % Adaptative step on additive noise ON
        opt_SAMP.signal     = x;

        % Execution de l'algorithme 

        x_hat7(:) = paSAMP(y,D,opt_SAMP);
        fprintf('paSAMP        =%1.2f\n',abs(x_hat7'*x)/(norm(x_hat7)*norm(x)));
        
        Perf_CorrNorm(7,cpt_param,posk)  = Perf_CorrNorm(7,cpt_param,posk)+abs(x_hat7'*x/(norm(x_hat7)*norm(x)));
        Perf_JCC(7,cpt_param,posk)      =Perf_JCC(7,cpt_param,posk) + JCC(abs(x),abs(x_hat7),posk);


        PerfSq_CorrNorm(7,cpt_param,posk)=PerfSq_CorrNorm(7,cpt_param,posk) + abs(x_hat7'*x/(norm(x_hat7)*norm(x)))*abs(x_hat7'*x/(norm(x_hat7)*norm(x))); % NormCorr ( metrique de performances de reconstruction de vecteur       
        PerfSq_JCC(7,cpt_param,posk)      =PerfSq_JCC(7,cpt_param,posk) + JCC(abs(x),abs(x_hat7),posk)*JCC(abs(x),abs(x_hat7),posk);



        disp(' ')
        
        
    end

end


            

end

Perf_JCC(:)=Perf_JCC./n_trial;
Perf_CorrNorm(:)=Perf_CorrNorm./n_trial;


Var_CorrNorm = (PerfSq_CorrNorm - (Perf_CorrNorm.*Perf_CorrNorm)./n_trial)./(n_trial-1);

Var_JCC      = (PerfSq_JCC - (Perf_JCC.*Perf_JCC)./n_trial)./(n_trial-1);

save_filename=['ResultsLAST/CROBIEN_Performance_vartheta' num2str(varp) 'compressedCorr_PR_' param_name '_dimy' num2str(dim_y) '_dims' num2str(dim_s) ...
                 '_ntrial' num2str(n_trial) '_varx' num2str(var_x)...
                '_varn' num2str(var_n) '_' flag_dico '_' flag_covp '_colength_5'];            

%  var_tmp = [param_vec Perf_CorrNorm Perf_JCC Var_CorrNorm Var_JCC];
% save([save_filename  '.dat'],'var_tmp','-ascii')
save([save_filename  '.mat'])

 end
 end

function D = dicoGen(dim_y,dim_s,dico_opt)

flag_dico = dico_opt.flag_dico;

if strcmp(flag_dico,'RandMod1')
    D=randn(dim_y,dim_s)+ 1i * randn(dim_y,dim_s);
    D(:)=D./abs(D);
elseif strcmp(flag_dico,'DoA')
    lambda=dico_opt.wavelength;
    %doa=(0:1:dim_s-1)'*(pi/2)/dim_s;           % between 0 and pi/2
    doa=(-pi/2)+(0:1:dim_s-1)'*(pi)/dim_s;      % between -pi/2 and pi/2
    D=exp(1i*2*pi*(1:1:dim_y)'*sin(doa')/lambda);
else
    error('Invalid choice for flag_dico')
end

normD=sqrt(sum(abs(D).^2,1));
D(:)=D./repmat(normD,dim_y,1);

%D
%abs(D)
%diag(D'*D)
%D'*D
%max(D'*D-eye(dim_s))
%pause

end



function [p] = genPhaseRandom(dim_y, m, icov, flag_covp)
% Model:

p=zeros(dim_y,1);

if strcmp(flag_covp,'complexNoise')

    % Warning: non-exact Von Mises sampling
    RE_NOISE  = mvnrnd((sqrt(1))*ones(dim_y, 1), 0.5*icov); % Warning: icov = covariance matrix (not precision)
	IM_NOISE  = mvnrnd(zeros(dim_y, 1), 0.5*icov);
    p(:)      = RE_NOISE + 1i.*IM_NOISE;
    p(:)      = angle(p(:)./abs(p(:)));
    
elseif strcmp(flag_covp,'1DVM')
    kappa = 2*icov(1,1);
    p(:) = circ_vmrnd(m(1), kappa, dim_y);
    
else % gaussian
    
    [U,Lambda]=svd(icov);
    p(:) = m + U*((1./sqrt(diag(Lambda))).*randn(dim_y,1));
    
end


end



function [m,icov] = paramPhaseRandom(dim_y,phase_opt)

m         = zeros(dim_y,1);
var_pt    = phase_opt.var_pt;
var_p1    = phase_opt.var_p1; 
drift     = phase_opt.drift;
flag_covp = phase_opt.flag_covp;

if strcmp(flag_covp,'full')
    one_vec       = ones(dim_y,1)/sqrt(dim_y);
    mattmp        = randn(dim_y,dim_y-1);
    mattmp(:)     = orth(mattmp-one_vec*one_vec'*mattmp);
    U             = [one_vec mattmp];
    
    Lambda        = rand(dim_y,1);
    min_Lambda    = min(Lambda);
    Lambda(:)     = Lambda/min_Lambda*(1./var_pt);
    Lambda(1)     = 1./var_p1;   % inverse variance associated to direction 'one_vec'
    
    %[U,Lambda]=svd(mattmp);
    %U(:,1)=1/sqrt(dim_y);
    
    icov       = U*diag(Lambda)*U';
elseif strcmp(flag_covp,'Markov')
    mattmp        = zeros(dim_y,3);
    mattmp(:,1)   = -drift/var_pt;
    mattmp(:,3)   = -drift/var_pt;
    mattmp(:,2)   = (1+drift^2)/var_pt;
    mattmp(1,2)   = drift^2/var_pt+1/var_p1;
    mattmp(end,2) = 1/var_pt;
    icov          = full(spdiags(mattmp,[-1 0 1],dim_y,dim_y));
elseif strcmp(flag_covp,'MarkovRand')
    mattmp        = zeros(dim_y,3);
    vartmp        = var_pt*rand(dim_y,1);
    mattmp(:,1)   = -drift./vartmp;
    mattmp(:,3)   = -drift./vartmp;
    mattmp(:,2)   = (1+drift^2)./vartmp;
    mattmp(1,2)   = drift^2/vartmp(1)+1/var_p1;
    mattmp(end,2) = 1/vartmp(end);
    icov          = full(spdiags(mattmp,[-1 0 1],dim_y,dim_y));    
elseif strcmp(flag_covp,'iid')    
    icov = eye(dim_y)/var_p1;
 elseif strcmp(flag_covp,'1DVM')
     icov = diag((1/var_pt)*ones(dim_y,1));    
else
    error('Invalid choice for flag_covp')
end

% [Lambda]=svd(icov);
% min_Lambda= min(Lambda)
% pause
end

function [JaccardIdx]=JCC(x,xhat,k)
% x is the true vector
% xhat the estimated one
% k is the number of sources

x_bin = zeros(length(x),1);
[~,locs1] = findpeaks(abs(x),'NPeaks',k);
x_bin(locs1)= 2;
% non-null coefficients equals to 2

xhat_bin = zeros(length(xhat),1);
xhat = xhat ./ max(xhat);
[~,locs2] = findpeaks(abs(xhat),'MinPeakProminence',0.1);
xhat_bin(locs2)= -1;
% forced non-null coefficient to -1


% So : x + xhat = 1 if well detected
%                -1 if false detection
%                 2 if missed

MISSED = sum((x_bin+xhat_bin)==2);
FD     = sum((x_bin+xhat_bin)==-1);
GD     = sum((x_bin+xhat_bin)==1);

JaccardIdx = GD / (GD+FD+MISSED) ;
if isnan(JaccardIdx)
    JaccardIdx=0;
end
end

