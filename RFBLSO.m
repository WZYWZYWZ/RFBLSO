clc;clear;
clc; clear; warning off
for ii = 1
addpath(genpath('./.'));
rng('default');
%% load data
load demodata.mat
%% RF
leaf = 5;
ntrees = 200;
disp('Training the tree bagger')
b = TreeBagger(ntrees, X,Y, 'Method','regression', 'oobvarimp','on', 'surrogate', 'on', 'minleaf',leaf,'FBoot',1);
disp('Estimate Output using tree bagger')
drawnow
%--------------------------------------------------------------------------
% Calculate the relative importance of the input variables
disp('Sorting importance into descending order')
weights=b.OOBPermutedVarDeltaError;
[B,iranked] = sort(weights,'descend');
%% Remove redundant brain regions
X(:,iranked(1,end))=[];
train_x = X;
train_y = Y;
test_x = X;
test_y = Y;
%% BLS
C1 = exp(-32:0.1:-25); % The range of regularization parameter selection.
s = .8;              %----s: the shrinkage parameter for enhancement nodes
best = 1;
result = [];
%%%%%%%%%%%%%%%%%%%%%%%
for NumFea= 1:3      %searching range for feature nodes  per window inaaaaaaa feature layer
    for NumWin=1:10      %searching range for number of windows in feature layer
        for NumEnhan=1:10  %searching range for enhancement nodes
            rand('state',1)     
            for i=1:NumWin
                WeightFea=2*rand(size(train_x,2)+1,NumFea)-1;
                %   b1=rand(size(train_x,2)+1,N1);  % sometimes use this may lead to better results, but not for sure!
                WF{i}=WeightFea;
            end                                                          %generating weight and bias matrix for each window in feature layer
            WeightEnhan=2*rand(NumWin*NumFea+1,NumEnhan)-1;
            %             WeightEnhan=rand(NumWin*NumFea+1,NumEnhan);    %You may choose one of the above initializing methods for weights connecting feature layer with enhancement layer
            fprintf(1, 'Fea. No.= %d, Win. No. =%d, Enhan. No. = %d\n', NumFea, NumWin, NumEnhan);
            [NetoutTrain,NetoutTest, Training_time,Testing_time, train_ERR,test_ERR,test_MAPE,MAE,MAE1] = bls_training(train_x,train_y,test_x,test_y,WF,WeightEnhan,s,C1,NumFea,NumWin);
            time =Training_time + Testing_time;% run time
            
            result = [result; NumFea NumWin NumEnhan test_ERR train_ERR MAE]; % recording all the searching reaults
            if best > test_ERR
                best = test_ERR;
                save('optimal.mat','test_ERR', 'train_ERR','NumFea', 'NumWin', 'NumEnhan','time');
            end
        end
    end
    toc
end
MAE = min(result(:,6)) % finall MAE
[x,y]= find (result(:,6) == min(result(:,6)))
cct=corrcoef(NetoutTest,test_y);
        cct=cct(2,1) % Pearson's r
end