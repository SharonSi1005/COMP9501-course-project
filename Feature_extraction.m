%% 
% Run LoadData.m from https://github.com/rdbraatz/data-drivenprediction-of-battery-cycle-life-before-capacity-degradation, 
% 
% then you can get the following .mat files.

load('batch_combined.mat');
load('bat_label.mat');
load('numBat.mat');
%% 
% Feature 1: 1C CC charging time, difference between cycle 10 and 100

chargingtime_1C_CC_10=zeros(numBat,1);
chargingtime_1C_CC_100=zeros(numBat,1);

F1=zeros(1,numBat);

for k = 1:numBat
    
    if k == 42
        
         chargingtime_1C_CC_10(k) = batch_combined(k).cycles(10).t(548)-batch_combined(k).cycles(10).t(408);
         chargingtime_1C_CC_100(k) = batch_combined(k).cycles(100).t(630)-batch_combined(k).cycles(100).t(509);
         
    else
        
         A = find((batch_combined(k).cycles(10).I < 1.01) & (batch_combined(k).cycles(10).I > 0.99));
         chargingtime_1C_CC_10(k) = batch_combined(k).cycles(10).t(A(end))-batch_combined(k).cycles(10).t(A(2));%第一个值可能不在1C-CC阶段
         B = find((batch_combined(k).cycles(100).I < 1.01) & (batch_combined(k).cycles(100).I > 0.99));
         chargingtime_1C_CC_100(k) = batch_combined(k).cycles(100).t(B(end))-batch_combined(k).cycles(100).t(B(2));
        
    end

    F1(k) = chargingtime_1C_CC_100(k)-chargingtime_1C_CC_10(k);
    
end
% rho_1=corrcoef(F1,bat_label);
%% 
% Feature 2 & 3: IC peak value and the corresponding votage, difference 
% between cycle 10 and 100

ICpeak_10=zeros(numBat,1);
ICpeak_100=zeros(numBat,1);

index10=zeros(numBat,1);
index100=zeros(numBat,1);

ICpeakV_10=zeros(numBat,1);
ICpeakV_100=zeros(numBat,1);

F2=zeros(1,numBat);
F3=zeros(1,numBat);

for k = 1:numBat
    
    [ICpeak_10(k),index10(k)] = max(abs(batch_combined(k).cycles(10).discharge_dQdV));
    [ICpeak_100(k),index100(k)] = max(abs(batch_combined(k).cycles(100).discharge_dQdV));
    
    F2(k) = ICpeak_100(k)-ICpeak_10(k);
    F3(k) = batch_combined(k).Vdlin(index100(k))-batch_combined(k).Vdlin(index10(k));
    
end


%% 
% Feature 4: Variance of deltaQ100-10(V)

Q_10=zeros(1000,numBat);
Q_100=zeros(1000,numBat);


for k = 1:numBat
    Q_10(:,k)=batch_combined(k).cycles(10).Qdlin;
    Q_100(:,k)=batch_combined(k).cycles(100).Qdlin;
    
end

F4=var(Q_100-Q_10);
%% 
% Feature 5: Variance of deltaTem100-2(V)

Tem_2=zeros(1000,numBat);
Tem_100=zeros(1000,numBat);


for k = 1:numBat
    Tem_2(:,k)=batch_combined(k).cycles(2).Tdlin;
    Tem_100(:,k)=batch_combined(k).cycles(100).Tdlin;
    
end

F5=var(Tem_100-Tem_2);
%% 
% Feature 6: Covering area of dTem/dV peak, difference between cycle 2 and 
% 100

deltaITempeak=zeros(numBat,999);
F6=zeros(1,numBat);

for k = 1:numBat
    
    dV = diff(batch_combined(k).Vdlin);

    Tem2 = smooth(batch_combined(k).cycles(2).Tdlin,13);
    Tem100 = smooth(batch_combined(k).cycles(100).Tdlin,13);
    
    dTem2 = diff(Tem2);
    dTemdV2 = dTem2./dV;

    dTem100 = diff(Tem100);
    dTemdV100 = dTem100./dV;
    
    deltaITempeak(k,:) = dTemdV100-dTemdV2;
    F6(k) = trapz(batch_combined(k).Vdlin(1:500),deltaITempeak(k,1:500)); %只计算峰值处的面积，去掉低电压部分的干扰

end
%% 
% Feature 7: Internal resistance(IR), difference between cycle 2 and 100

IR_2=zeros(1,numBat);
IR_100=zeros(1,numBat);

for k = 1:numBat
    
    IR_2(k) = batch_combined(k).summary.IR(2);
    IR_100(k) = batch_combined(k).summary.IR(100);
    
end

F7 = IR_100-IR_2;
%%
batFeatureData = [F1;F2;F3;F4;F5;F6;F7];
remove_ind = [48,49,62,89,90];
batFeatureData(:,remove_ind) = [];
batFeatureDataNormal = normalize(batFeatureData,2,'range');
bat_label(remove_ind,:) = [];
%%

all_ind = linspace(1,119,119);

train_ind = randsample(119,80); % training set index
Xtrain = batFeatureDataNormal(:,train_ind);
ytrain = bat_label(train_ind)';

istrain_ind = ismember(all_ind,train_ind);
test_ind = all_ind(~istrain_ind); % testing set index
Xtest = batFeatureDataNormal(:,test_ind);
ytest = bat_label(test_ind)';