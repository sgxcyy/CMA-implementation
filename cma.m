function [a1,a2,a3] = cma( )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
addpath cache_fts/
addpath libsvm/
load caltran_gist;
load caltran_dataset_labels;
feature_type = 'gist';
start_index = round(rand()*4000);
%Parameters
expt = config_caltran(feature_type, start_index);
features = data.features;
labels = labels';
fnames = names;
% Remove NANs
[r,c] = find(isnan(features));
ind = find(~ismember(1:size(features,1), r));
features = features(ind,:);
features = double(features);
labels = labels(ind);
labels = labels(:);
fnames = fnames(ind);
%NormalizeData
features = NormalizeData(features', expt.norm_type)';
% Create training and test split
tr_ind = expt.start:(expt.ns+expt.start);
T = min(expt.Tmax,size(features, 1)); % can't go beyond the number of data points available
te_ind = (expt.ns+expt.start+1):(expt.ns+expt.start+1+T);

Xs = features(tr_ind, :);
Ys = labels(tr_ind);
fnames_s = fnames(tr_ind);
Xt = features(te_ind,:);
Yt = labels(te_ind);
fnames_t = fnames(te_ind);

T = size(Xt, 1);
[~,Ss,Vs] = svd(Xs);
Us = Vs(:,1:expt.dim);
Q = [Us, null(Us')];
Pt = Us; % Initialize
m = min(size(Ss,1),expt.dim);
S = diag(Ss(1:m,1:m));
S((m+1):expt.dim) = 0;
mu = mean(Xs)';
nprev = size(Xs,1);
Xt_cgfk = zeros(size(Xt));
Xt_csa = zeros(size(Xt));
fprintf('[Adapt] Continuous Manifold Adaptation (CMA) '); 
for i = expt.block_size:expt.block_size:T
    ind = (i-expt.block_size+1):i;
    
    [Pt, S, mu, nprev] = sklm(Xt(ind,:)', Pt, S, mu, nprev, expt.alpha, expt.dim);
    if expt.fast_mode
        G = fastGFK(Q, Pt);
    else
        G = GFK(Q, Pt);
    end
    Xt_cgfk(ind,:) = Xt(ind,:) * G;
    Xt_csa(ind,:) = Xt(ind,:) * (Pt*Pt');
    if mod(i/expt.block_size, 10) == 0
        fprintf('.');
    end

end
fprintf('\n');
model = train(Ys, sparse(Xs), sprintf('-c %d -q', expt.C));
[pred,a1,pr] = predict(Yt, sparse(Xt), model,'-q');
[pred,a2,pr] = predict(Yt, sparse(Xt_csa), model,'-q');
[pred,a3,pr] = predict(Yt, sparse(Xt_cgfk), model,'-q');
a1 =(a1(1));
a2 =(a2(1));
a3 =(a3(1));
end

