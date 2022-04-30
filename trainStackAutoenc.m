rng('default')

maxEpochs=3000;

hiddenSize1 = 20;
autoenc1 = trainAutoencoder(Xtrain,hiddenSize1, ...
    'MaxEpochs',maxEpochs,...
    'SparsityRegularization',0, ...
    'ScaleData', false);

feat1 = encode(autoenc1,Xtrain);
% Xrecon = predict(autoenc1,Xtrain);


hiddenSize2 = 10;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',maxEpochs,...
    'SparsityRegularization',0, ...
    'ScaleData', false);

feat2 = encode(autoenc2,feat1);

hiddenSize3 = 8;
autoenc3 = trainAutoencoder(feat2,hiddenSize3, ...
    'MaxEpochs',maxEpochs,...
    'SparsityRegularization',0, ...
    'ScaleData', false);

feat3 = encode(autoenc3,feat2);



hiddenSize4 = 4;
autoenc4 = trainAutoencoder(feat3,hiddenSize4, ...
    'MaxEpochs',maxEpochs,...
    'SparsityRegularization',0, ...
    'ScaleData', false);

feat4 = encode(autoenc4,feat3);

net1 = feedforwardnet(3);

net1 = train(net1,feat4,ytrain);
view(net1)



stackednet = stack(autoenc1,autoenc2,autoenc3,autoenc4,net1);
view(stackednet)


stackednet.trainParam.epochs = 1000;


stackednet.divideFcn = 'divideind';
stackednet.divideParam.trainInd = [1:60];
stackednet.divideParam.valInd = [61:80];
stackednet.divideParam.testInd =[];
stackednet = train(stackednet,Xtrain,ytrain);

%% model evaluation

%training set
y1train=stackednet(Xtrain(:,1:60));
MSE1train = mean((y1train-ytrain(1:60)).^2);
RMSE1train = sqrt(MSE1train);
MPE1train = mean(abs(y1train-ytrain(1:60))./ytrain(1:60))*100; % mean percent error

%validation set
y1val=stackednet(Xtrain(:,61:80));
MSE1val = mean((y1val-ytrain(61:80)).^2);
RMSE1val = sqrt(MSE1val);
MPE1val = mean(abs(y1val-ytrain(61:80))./ytrain(61:80))*100; % mean percent error

y2 = stackednet(Xtest);

%testing set
MSE2 = mean((y2-ytest).^2);
RMSE2 = sqrt(MSE2);
MPE2 = mean(abs(y2-ytest)./ytest)*100; % mean percent error