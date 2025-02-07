clc;clear;
clc; clear; warning off
for ii = 1
addpath(genpath('./.'));
rng('default');
%%
% load data2000_2000.mat;
% train_x1 = train_x;
% train_y1 = train_y;
% test_x1 = test_x;
% test_y1 = test_y;
% load NC.mat;
% train_x2 = train_x;
% train_y2 = train_y;
% test_x2 = test_x;
% test_y2 = test_y;
% %%
% train_x = train_x2;
% train_y = train_y2;
% test_x = test_x1;
% test_y = test_y1;
%%
load NC_94.mat;
train_x = newdata(:,1:94);
train_y = newdata(:,95);
load ADNCMCI.mat;
test_x = NC(:,1:94);
test_y = NC(:,95);
%%
% load NC_94.mat;
% train_x = newdata(:,1:94);
% train_y = newdata(:,95);
% load ADMNICI.mat;
% test_x = NC(:,1:94);
% test_y = NC(:,95);
%%
% C1 = exp(-32:0.1:-25);
C1 = 2^-30;
s = .8;              %----s: the shrinkage parameter for enhancement nodes
best = 1;
result = [];
%%%%%%%%%%%%%%%%%%%%%%%
for NumFea= 12:22      %searching range for feature nodes  per window inaaaaaaa feature layer
    for NumWin=12:22      %searching range for number of windows in feature layer
        for NumEnhan=25:40  %searching range for enhancement nodes
% for NumFea= 15      %searching range for feature nodes  per window inaaaaaaa feature layer
%     for NumWin=16      %searching range for number of windows in feature layer
%         for NumEnhan=40  %           
                              clc;
            rand('state',1)     
            for i=1:NumWin
                WeightFea=2*rand(size(train_x,2)+1,NumFea)-1;
                %   b1=rand(size(train_x,2)+1,N1);  % sometimes use this may lead to better results, but not for sure!
                WF{i}=WeightFea;
            end                                                          %generating weight and bias matrix for each window in feature layer
            %             if NumFea*NumWin>=NumEnhan
            %                 WeightEnhan=orth(2*rand(NumWin*NumFea+1,NumEnhan)-1);
            %             else
            %                 WeightEnhan=orth(2*rand(NumWin*NumFea+1,NumEnhan)'-1)';
            %             end
            WeightEnhan=2*rand(NumWin*NumFea+1,NumEnhan)-1;
            %             WeightEnhan=rand(NumWin*NumFea+1,NumEnhan);    %You may choose one of the above initializing methods for weights connecting feature layer with enhancement layer
            fprintf(1, 'Fea. No.= %d, Win. No. =%d, Enhan. No. = %d\n', NumFea, NumWin, NumEnhan);
            [NetoutTrain,NetoutTest, Training_time,Testing_time, train_ERR,test_ERR,test_MAPE,MAE,MAE1] = bls_training(train_x,train_y,test_x,test_y,WF,WeightEnhan,s,C1,NumFea,NumWin);
            time =Training_time + Testing_time;
            
            result = [result; NumFea NumWin NumEnhan test_ERR train_ERR MAE]; % recording all the searching reaults
            if best > test_ERR
                best = test_ERR;
                save('optimal.mat','test_ERR', 'train_ERR','NumFea', 'NumWin', 'NumEnhan','time');
            end
%             clearvars -except best NumFea NumWin NumEnhan train_x train_y test_x test_y   C1 s result NetoutTest
        end
    end
    toc
end
% % end
abc = min(result(:,6))
[x,y]= find (result(:,6) == min(result(:,6)))
cct=corrcoef(NetoutTest,test_y);
        cct=cct(2,1)
abc(ii) = min(result(:,6))
% [x,y]= find (result(:,6) == min(result(:,6)))  
%  clearvars -except abc
end
%%
disp('Create a scatter Diagram')
% plot the 1:1 line
x = test_y;
y = NetoutTest;
plot(x,x,'LineWidth',3);
hold on
scatter(x,y,'filled');
hold off
grid on
set(gca,'FontSize',12)
xlabel('Actual','FontSize',12)
ylabel('Estimated','FontSize',12)
% title(['Training Dataset, R^2=' num2str(cct^2,2)],'FontSize',30)
title(['r=' num2str(cct)],'FontSize',14)
drawnow
%%
% terr = NetoutTest-test_y;
% tra = NetoutTrain - train_y;
% atrain = [train,tra];
% atest = [test,terr];







