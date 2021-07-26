clear;
clc;
close all;

run_index=[1:10];
 feature_weight = zeros(10,40);
for i=1:10
[train,text]=xlsread(['..\result\multi run\run',num2str(run_index(i)),'\10%split.xlsx'],'train dataset');
[test,text]=xlsread(['..\result\multi run\run',num2str(run_index(i)),'\10%split.xlsx'],'test dataset');
load(['..\result\multi run\run',num2str(run_index(i)),'\feature weight10%.mat'])
final_ck(:,all(final_ck==0,1)) = [];
feature_weight(i,:)=final_ck(:,end);
end

ave_feature_weight= mean(feature_weight,1);

[labelednum,~]=size(train);
[unlabelednum,~]=size(test);
class_test=test(:,1);
feature_train_test=[train(:,2:end);test(:,2:end)];

train_label=train(:,1);
test_label=test(:,1);
test_feature =zeros(size(feature_train_test,1),40);

for j=1:40
test_feature(:,j) = feature_train_test(:,j)*ave_feature_weight(j);
end

new_Y = tsne(test_feature);



%% two plot
load(['..\result\multi run\run',num2str(run_index(i)),'\finallabel10%.mat'])
shallow_index=177:444;
deep_index=44:176;
new_group1=[new_Y(shallow_index,2);new_Y(deep_index,2)];
new_group2=[new_Y(shallow_index,1);new_Y(deep_index,1)];
label=[2*ones(268,1);ones(133,1)];
gscatter(new_group2,new_group1,label)


figure
predict_label=x_valid(43:end);
predict_label = sign(predict_label);
[deep_row,deep_col]=find(predict_label==1);
[shallow_row,shallow_col]=find(predict_label==-1);
new_group3=[new_Y(shallow_row,2);new_Y(deep_row,2)];
new_group4=[new_Y(shallow_row,1);new_Y(deep_row,1)];
label=[2*ones(length(shallow_row),1);ones(length(deep_row),1)];
gscatter(new_group4,new_group3,label)


