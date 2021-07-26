clear
clc
cvx_begin quiet

[data,text] = xlsread('final_dataset_for_training。xlsx','standardized features');
label = data(:,1);
q=9; % the number of folders
indices = crossvalind('Kfold',label,q);

%% graph setting
num_feature=40;
S_lower=0;
S_upper=100;
c_k_initially_set = 1;
tol_set=0.01;
tol_set_pg=0.01;
sigma_index=1;
sigma=1;
alpha=1;
%% inverse q folder cross validation
final_CER=zeros(q,1);
for i = 1:q
    train_index = (indices == i); 
    test_index = ~train_index;
    train_feature=data(train_index,2:end);
    test_feature=data(test_index,2:end);
    train_label = label(train_index);
    test_label = label(test_index);
    
    class_test = test_label;
    labelednum = length(find(train_index~=0));
    unlabelednum = length(find(test_index~=0));
    feature_train_test =[train_feature;test_feature];
    initial_label=[train_label;zeros(unlabelednum,1)];
    initial_label_index = logical([ones(labelednum,1); zeros(unlabelednum, 1)]);
    class_train_test=[train_label;test_label];

    x_known=[train_label;ones(length(class_train_test)-length(train_label),1)*0];
   

    [ x_valid, class_SDP_temp, SDP_error, ck_A,final_ck] = ...
    sdp_binary_GU_oao_L_constant_sign_o_norm_replace_GLR_GTV_L_norm( class_test, ...
    c_k_initially_set, tol_set, tol_set_pg, S_lower, S_upper, ...
    feature_train_test, initial_label, initial_label_index, class_train_test,alpha,train_label,sigma_index);

 %% check result
    diff_label = sign(x_valid)-class_train_test;
    classification_error_rate = size(find(diff_label~=0),1)*size(find(diff_label~=0),2)/size(class_test,1);
    final_CER(q)=classification_error_rate;

end