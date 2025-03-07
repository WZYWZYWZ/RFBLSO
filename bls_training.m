function             [NetoutTrain,NetoutTest, Training_time,Testing_time, train_ERR,test_ERR,test_MAPE,MAE,MAE1] = bls_training(train_x,train_y,test_x,test_y,WF,WeightEnhan,s,C,NumFea,NumWin)

tic
y=zeros(size(train_x,1),NumWin*NumFea);
H1 = [train_x,  0.1 * ones(size(train_x,1),1)];
for i=1:NumWin
    WeightFea=WF{i};
    A1 = H1 * WeightFea;A1 = mapminmax(A1);%Map matrix row minimum and maximum values to [-1 1].
    clear WeightFea;
    WeightFeaSparse  = sparse_bls(A1,H1,1e-3,50)';
    WFSparse{i}=WeightFeaSparse;
    
    T1 = H1 * WeightFeaSparse;
    [T1,ps1]  =  mapminmax(T1',0,1);
    T1 = T1';
    
    ps(i)=ps1;
    y(:,NumFea*(i-1)+1:NumFea*i)=T1;
end

clear H1;
clear T1;
H2 = [y,  0.1 * ones(size(y,1),1)];
T2 = H2 * WeightEnhan;
%%
T2 = tansig(T2);
%%
T3=[y T2];
[beta_opt,C_opt,LOO] = lOO(T3,train_y,C); % LOO
clear H2;
clear T2;
WeightTop = (T3'  *  T3+eye(size(T3',1)) * (C)) \ ( T3'  *  train_y);
%%
Training_time = toc;
disp('Training has been finished!');
disp(['The Total Training Time is : ', num2str(Training_time), ' seconds' ]);
NetoutTrain = T3 * WeightTop;
clear T3;

RMSE =  sqrt(sum((NetoutTrain-train_y).^2)/size(train_y,1));
MAPE = sum(abs(NetoutTrain-train_y))/mean(train_y)/size(train_y,1);
MAE1 = sum(abs(NetoutTrain-train_y))/size(train_y,1);
train_ERR = RMSE;
fprintf(1, 'Training RMSE is : %e, Training MAPE is: %e\n', RMSE, MAPE);
tic;
yy1=zeros(size(test_x,1),NumWin*NumFea);
HH1 = [test_x .1 * ones(size(test_x,1),1)];

for i=1:NumWin
    WeightFeaSparse=WFSparse{i};ps1=ps(i);
    TT1 = HH1 * WeightFeaSparse;
    TT1  =  mapminmax('apply',TT1',ps1)';
    
    clear WeightFeaSparse; clear ps1;
    %yy1=[yy1 TT1];
    yy1(:,NumFea*(i-1)+1:NumFea*i)=TT1;
end
clear TT1;clear HH1;
HH2 = [yy1 .1 * ones(size(yy1,1),1)];
% TT2 = tansig(HH2 * b2 * l2);0
%%
TT2 = tansig(HH2 * WeightEnhan);
%%
TT3=[yy1 TT2];
clear HH2;clear b2;clear TT2;

NetoutTest = TT3 * WeightTop;

RMSE = sqrt(sum((NetoutTest-test_y).^2)/size(test_y,1));
%      MSE = sum((x-test_y).^2)/size(test_y,1);
% MAE = mean(abs(NetoutTest-test_y));
MAE = sum(abs(NetoutTest-test_y))/size(test_y,1);
MAPE = sum(abs(NetoutTest-test_y))/mean(test_y)/size(test_y,1);

clear TT3;
test_ERR = RMSE;
test_MAPE = MAPE;
%% Calculate the testing accuracy
Testing_time = toc;
disp('Testing has been finished!');
disp(['The Total Testing Time is : ', num2str(Testing_time), ' seconds' ]);
fprintf(1, 'Testing RMSE is : %e, Testing MAPE is: %e\n', RMSE, MAPE);
% fprintf(1, 'Testing MSE is : %e, Testing MAPE is: %e\n', MSE, MAPE);
