clear
clc

%% data standardization
% [data,~]=xlsread('final_dataset_for_training.xlsx','original features');
% type = ['standard'];
% normalized_train_set = StatisticalNormaliz(data,type);
% total_num_deep=147;
% deep_label=ones(total_num_deep,1);
% deep_data=[deep_label,normalized_train_set(1:total_num_deep,5:end)];
% total_num_shallow=297;
% shallow_label=-1*ones(total_num_shallow,1);
% shallow_data=[shallow_label,normalized_train_set(total_num_deep+1:end,5:end)];
% xlswrite('final_dataset_for_training.xlsx',deep_data,'deep standardized')
% xlswrite('final_dataset_for_training.xlsx',shallow_data,'shallow standardized')

%% split training and testing set with training ratio 10% to 90%
[deep_sta,~]=xlsread('final_dataset_for_training.xlsx','deep standardized');
overall_deep_feature = deep_sta(:,2:end);
[shallow_sta,text]=xlsread('final_dataset_for_training.xlsx','shallow standardized');
overall_shallow_feature = shallow_sta(:,2:end);
total_num_deep=147;
total_num_shallow=297;
training_ratio=[10:10:90];
for i=1:9

training_ratio_ratio=training_ratio(i)/100; % training ratio
train_num_deep_class=fix(training_ratio_ratio*total_num_deep);
test_num_deep_class = total_num_deep - train_num_deep_class;
train_num_shallow_class=fix(training_ratio_ratio*total_num_shallow);
test_num_shallow_class = total_num_shallow-train_num_shallow_class;
deep_class_all_index=[1:total_num_deep]';
shallow_class_all_index=[1:total_num_shallow]';
shuffle_train_deep_classindex=shuffle(deep_class_all_index);
shuffle_train_deep_classindex = shuffle_train_deep_classindex(1:train_num_deep_class);
shuffle_train_shallow_classindex=shuffle(shallow_class_all_index);
shuffle_train_shallow_classindex = shuffle_train_shallow_classindex(1:train_num_shallow_class);

train_set = [overall_deep_feature(shuffle_train_deep_classindex,:);overall_shallow_feature(shuffle_train_shallow_classindex,:)];
train_label = [ones(train_num_deep_class,1);(-1)*ones(train_num_shallow_class,1)];
train_whole_data=[train_label,train_set];
% 
test_deep_class_index = setdiff(deep_class_all_index, shuffle_train_deep_classindex);
test_shallow_class_index = setdiff(shallow_class_all_index, shuffle_train_shallow_classindex);
test_set = [overall_deep_feature(test_deep_class_index,:);overall_shallow_feature(test_shallow_class_index,:)];
test_label = [ones(test_num_deep_class,1);(-1)*ones(test_num_shallow_class,1)];
test_whole_data=[test_label,test_set];

xlswrite([num2str(training_ratio(i)),'%split.xlsx'],train_whole_data,'train dataset')
xlswrite([num2str(training_ratio(i)),'%split.xlsx'],test_whole_data,'test dataset')
end


