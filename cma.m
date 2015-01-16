function y = cma( )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
addpath cache_fts/
load caltran_gist;
load caltran_dataset_labels;
feature_type = 'gist';
start_index = 350;
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
Ps = Vs(:,1:expt.dim);
end

