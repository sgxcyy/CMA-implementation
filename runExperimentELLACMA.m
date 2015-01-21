%%
% Do an active task selection experiment on the landmine data
%
% Copyright (C) Paul Ruvolo and Eric Eaton 2013
%
% This file is part of ELLA.
%
% ELLA is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% ELLA is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with ELLA.  If not, see <http://www.gnu.org/licenses/>.
function y = runExperimentActiveTask()
	clear all;
    useLogistic = true;
%	load('Datasets/landminedata','feature','label');
%   load('MITfeatures','feature','test');
    load('caltran_gist');
    feature_type = 'gist';
    start_index = 350;
    expt = config_caltran(feature_type, start_index);
    features = data.features;
    labels = data.labels';
    [r,c] = find(isnan(features));
    ind = find(~ismember(1:size(features,1), r));
    features = features(ind,:);
    features = double(features);
    labels = labels(ind);
    labels = labels(:);
    features = NormalizeData(features', 'l1_zscore')';
    T = 20;
%     [Us,Ss,Vs] = svd(features);
%      features = features*Vs(:,1:100);
    Nt = floor(length(labels)/T);
    feature = cell(T,1);
    label = cell(T,1);
    Xs = features(1:Nt,:);
    Xt = features(Nt+1:5412,:);
    expt.block_size = 3;
%     [Xt_cgfk, Xt_csa] = cma(expt, Xs, Xt);
%     features = [Xs;Xt_csa];
    for i = 1:T
        feature{i}= features((i-1)*Nt+1:Nt*i,:);
        label{i} = labels((i-1)*Nt+1:Nt*i,:);
    end
%    T = length(feature);
%     for t =1:T
%         feature{t} = [feature{t};test{t}];
%     end
%	learningCurves = zeros(T);
	for t = 1 : T
	    feature{t}(:,end+1) = 1;
%         label{t} = label{t}.*2-1;
	end
%     label = cell(T,1);
%     for t =1:T
%         label{t}=zeros(size(feature{t},1),T);
%         label{t}(:,t)=label{t}(:,t)+1;
%     end
    
	sizef=zeros(T,1);

	X = cell(T,1);
	Xtest = cell(T,1);
	Y = cell(T,1);
	Ytest = cell(T,1);
  %  theta1= cell(1,T);
	for t = 1 : T
        sizef(t)=size(feature{t},1);
	    r = randperm(sizef(t));
	    traininds = r(1:floor(length(r)/2));
	    testinds = r(floor(length(r)/2)+1:end);
	    X{t} = feature{t}(traininds,:);
	    Xtest{t} = feature{t}(testinds,:);
% 	    Y{t} = label{t}(traininds);
% 	    Ytest{t} = label{t}(testinds);
        Y{t} = label{t}(traininds,:);
	    Ytest{t} = label{t}(testinds,:);
%         idx1 = find(Y{t}==1);
%         idx2 = find(Y{t}==-1);
%         idx = [idx2(randperm(floor(length(idx2)/1))); idx1];
%         X{t} = X{t}(idx,:);
%         Y{t} = Y{t}(idx);
    end
    
  
 % testDNN;
    
    d = size(X{1},2);
%     
    learningCurves = zeros(T);
    model = initModelELLA(struct('k',2,...
	    			     'd',d,...
	    			     'mu',exp(-12),...
	    			     'lambda',exp(-10),...
	    			     'ridgeTerm',exp(-5),...
	    			     'initializeWithFirstKTasks',true,...
	    			     'useLogistic',useLogistic,...
	    			     'lastFeatureIsABiasTerm',true));
	learned = logical(zeros(length(Y),1));
        adb = zeros(T,T);
	unlearned = find(~learned);
	for t = 1 : T
%         if t>1
%            X{t} = [X{t};X{t-1}];
%            Y{t} = [Y{t};Y{t-1}];
%         end 
	    % change the last input to 1 for random, 2 for InfoMax, 3 for Diversity, 4 for Diversity++
	    idx = selectTaskELLA(model,{X{unlearned}},{Y{unlearned}},2);
	    model = addTaskELLA(model,X{unlearned(idx)},Y{unlearned(idx)},unlearned(idx));
	    learned(unlearned(idx)) = true;
	    unlearned = find(~learned);
	    % encode the unlearned tasks
	    for tprime = 1 : length(unlearned)
		model = addTaskELLA(model,X{unlearned(tprime)},Y{unlearned(tprime)},unlearned(tprime),true);
	    end
	    for tprime = 1 : T
	    	preds{t,tprime} = predictELLA(model,Xtest{tprime},tprime);
            %AUC
		  %  learningCurves(t,tprime) = roc(preds{t,tprime},Ytest{tprime});
            %AP
            preds{t,tprime}(preds{t,tprime}>=0.5)=1;
            preds{t,tprime}(preds{t,tprime}<0.5)=-1;
            f = find(preds{t,tprime}~=Ytest{tprime});
            learningCurves(t,tprime) = 1-length(f)/length(Ytest{tprime});
        end

    end


	y = mean(learningCurves,2);
%    plot(y','+');
end
