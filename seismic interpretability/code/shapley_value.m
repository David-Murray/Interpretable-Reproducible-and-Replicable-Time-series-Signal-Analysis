clear;
clc;
close all;

 split=[10:10:70];
for i=1
  
[train,text]=xlsread([num2str(split(i)),'%split.xlsx'],'train dataset');
[test,text]=xlsread([num2str(split(i)),'%split.xlsx'],'test dataset');

[labelednum,~]=size(train);
[unlabelednum,~]=size(test);
class_test=test(:,1);
feature_train_test=[train(:,2:end);test(:,2:end)];
% test_feature = test(:,2:end);
train_label=train(:,1);
test_label=test(:,1);
test_feature =zeros(size(test,1),40);

end

maxAmp = test(:,2);
spCen = test(:,3);
rmsA = test(:,4);
maxPFA = test(:,5);
maxFA = test(:,6);
maxPF_FA = test(:,7);
Energy = test(:,8);
rect = test(:,9);
azmth = test(:,10);
indAngl = test(:,11);

Dip = test(:,12);
DipRec = test(:,13);
ccD = test(:,14);
ccS = test(:,15);
xotsu =test(:,16);
ccnAb2D = test(:,17);
ccnAb2S = test(:,18);
ypicD = test(:,19);
ypicS = test(:,20);
ccnRel2D = test(:,21);

ccnRel2S = test(:,22);
ccAb2D = test(:,23);
ccAb2S = test(:,24);
ccRel2D = test(:,25);
ccRel2S = test(:,26);
xp2D = test(:,27);
xp2S = test(:,28);
xpp2D = test(:,29);
xpp2S = test(:,30);
semD = test(:,31);

semS = test(:,32);
skwnss = test(:,33);
semblanceD = test(:,34);
semblanceS = test(:,35);
envelopD = test(:,36);
envelopS = test(:,37);
udD = test(:,38);
udS = test(:,39);
ud2D = test(:,40);
ud2S = test(:,41);

%% shapley values
feature_weight_class1=zeros(40,401);
feature_weight_class2=zeros(40,401);
for j=1:401
    disp(j)
class=cellstr(num2str(test_label));

T = table(maxAmp,spCen,rmsA,maxPFA,maxFA,maxPF_FA,Energy,rect,azmth,indAngl,Dip,DipRec,ccD,ccS,xotsu,ccnAb2D,ccnAb2S,ypicD,ypicS,ccnRel2D,ccnRel2S,ccAb2D,ccAb2S,ccRel2D,ccRel2S,xp2D,xp2S,xpp2D,xpp2S,semD,semS,skwnss,semblanceD,semblanceS,envelopD,envelopS,udD,udS,ud2D,ud2S,class);
blackbox = fitcecoc(T,'class', ...
    'PredictorNames',T.Properties.VariableNames(1:40), ...
    'ClassNames',{'1' '-1'});
queryPoint = T(j,:);
explainer = shapley(blackbox,'QueryPoint',queryPoint);
% explainer.ShapleyValues;
class1_shapyle_value = table2array(explainer.ShapleyValues(:,2));
class2_shapyle_value = table2array(explainer.ShapleyValues(:,3));
feature_weight_class1(:,j)=class1_shapyle_value;
feature_weight_class2(:,j)=class2_shapyle_value;
end

% save('shapley_value_class1.mat','feature_weight_class1')
% save('shapley_value_class2.mat','feature_weight_class2')