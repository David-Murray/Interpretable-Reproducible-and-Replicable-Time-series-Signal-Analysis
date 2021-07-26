clear;
clc;
close all;
cvx_begin quiet
diary myDiaryFile
global labelednum unlabelednum 
 split_ratio=[10:10:90];
for i=1:9
[train,text]=xlsread([num2str(split_ratio(i)),'%split.xlsx'],'train dataset');
[test,text]=xlsread([num2str(split_ratio(i)),'%split.xlsx'],'test dataset');

[labelednum,~]=size(train);
[unlabelednum,~]=size(test);
disp(unlabelednum)
disp('=========start========');
num_feature=40;
S_lower=0;
S_upper=100;
c_k_initially_set = 1;
tol_set=0.01;
tol_set_pg=0.01;


class_test=test(:,1);
feature_train_test=[train(:,2:end);test(:,2:end)];
train_label=train(:,1);
test_label=test(:,1);
initial_label_index = logical([ones(labelednum,1); zeros(unlabelednum, 1)]);
class_train_test=[train_label;test_label];
x_known=[train_label;ones(length(class_train_test)-length(train_label),1)*0];
   
[ x_valid, class_SDP_temp, SDP_error, ck_A,final_ck] = ...
    sdp_binary_GU_oao_L_constant_sign_o_norm_replace_GLR_GTV_L_norm( class_test, ...
    c_k_initially_set, tol_set, tol_set_pg, S_upper, ...
    feature_train_test, initial_label_index, class_train_test,train_label);

save(['feature weight',num2str(split_ratio(i)),'%.mat'],'final_ck');
save(['finallabel',num2str(split_ratio(i)),'%.mat'],'x_valid');
end
