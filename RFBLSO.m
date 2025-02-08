clc;clear;
clc; clear; warning off
for ii = 1
addpath(genpath('./.'));
rng('default');
%%
%%
% 加载Matlab提供的测试数据——使用1985年汽车进口量数据库，其中包含205个样本数据，25个自变量和1个因变量
load data2005;
Y = X(:,1);
X = X(:,2:end);
% isCategorical = [zeros(15,1);ones(size(X,2)-15,1)]; % Categorical variable flag

%% 训练随机森林，TreeBagger使用内容，以及设置随机森林参数
tic
leaf = 5;
ntrees = 200;
fboot = 1;
disp('Training the tree bagger')
b = TreeBagger(ntrees, X,Y, 'Method','regression', 'oobvarimp','on', 'surrogate', 'on', 'minleaf',leaf,'FBoot',fboot);
toc

%% 使用训练好的模型进行预测
% 这里没有单独设置测试数据集合，如果进行真正的预测性能测试，使用未加入至模型训练的数据进行预测测试。
disp('Estimate Output using tree bagger')
x = Y;
y = predict(b, X);
toc

% calculate the training data correlation coefficient
% 计算相关系数
cct=corrcoef(x,y);
cct=cct(2,1);

% Create a scatter Diagram
disp('Create a scatter Diagram')

% plot the 1:1 line
plot(x,x,'LineWidth',3);

hold on
scatter(x,y,'filled');
hold off
grid on

set(gca,'FontSize',18)
xlabel('Actual','FontSize',25)
ylabel('Estimated','FontSize',25)
title(['Training Dataset, R^2=' num2str(cct^2,2)],'FontSize',30)

drawnow

fn='ScatterDiagram';
fnpng=[fn,'.png'];
print('-dpng',fnpng);

%--------------------------------------------------------------------------
% Calculate the relative importance of the input variables
tic
disp('Sorting importance into descending order')
weights=b.OOBPermutedVarDeltaError;
[B,iranked] = sort(weights,'descend');
toc

%--------------------------------------------------------------------------
disp(['Plotting a horizontal bar graph of sorted labeled weights.']) 

%--------------------------------------------------------------------------
figure
barh(weights(iranked),'g');
xlabel('Variable Importance','FontSize',30,'Interpreter','latex');
ylabel('Variable Rank','FontSize',30,'Interpreter','latex');
title(...
    ['Relative Importance of Inputs in estimating Redshift'],...
    'FontSize',17,'Interpreter','latex'...
    );
hold on
barh(weights(iranked(1:10)),'y');
barh(weights(iranked(1:5)),'r');

%--------------------------------------------------------------------------
grid on 
xt = get(gca,'XTick');    
xt_spacing=unique(diff(xt));
xt_spacing=xt_spacing(1);    
yt = get(gca,'YTick');    
ylim([0.25 length(weights)+0.75]);
xl=xlim;
xlim([0 2.5*max(weights)]);

%--------------------------------------------------------------------------
% Add text labels to each bar
for ii=1:length(weights)
    text(...
        max([0 weights(iranked(ii))+0.02*max(weights)]),ii,...
        ['Column ' num2str(iranked(ii))],'Interpreter','latex','FontSize',11);
end

%--------------------------------------------------------------------------
set(gca,'FontSize',16)
set(gca,'XTick',0:2*xt_spacing:1.1*max(xl));
set(gca,'YTick',yt);
set(gca,'TickDir','out');
set(gca, 'ydir', 'reverse' )
set(gca,'LineWidth',2);   
drawnow

%--------------------------------------------------------------------------
fn='RelativeImportanceInputs';
fnpng=[fn,'.png'];
print('-dpng',fnpng);

%--------------------------------------------------------------------------
% Ploting how weights change with variable rank
disp('Ploting out of bag error versus the number of grown trees')

figure
plot(b.oobError,'LineWidth',2);
xlabel('Number of Trees','FontSize',30)
ylabel('Out of Bag Error','FontSize',30)
title('Out of Bag Error','FontSize',30)
set(gca,'FontSize',16)
set(gca,'LineWidth',2);   
grid on
drawnow
fn='EroorAsFunctionOfForestSize';
fnpng=[fn,'.png'];
print('-dpng',fnpng);
%%
newdata=X(1:B,end);
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
C1 = exp(-32:0.1:-25);
% C1 = 2^-30;
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







