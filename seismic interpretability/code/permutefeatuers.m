
clear;
clc;
close all;
cvx_begin quiet
diary myDiaryFile
global labelednum unlabelednum 
split=[10:10:70];
i=1;
   
[train,text]=xlsread([num2str(split(i)),'%split.xlsx'],'train dataset');
[test,text]=xlsread([num2str(split(i)),'%split.xlsx'],'test dataset');
load('feature weight10%.mat')
load('shapley_value_class1.mat')
paper_feature_weight=[0.080,0.445,0.286,0.278,0.27,0.092,0.0501,0.358,0.0901,0.052,0.261,0.205,0.254,0.076,0.167,0.262,0.122,0.101,0.094,0,0.081,0,0,0,0,0.088,0.214,0,0,0.21,0,0.13,0.132,0.050,0.245,0.234,0.127,0.111,0,0];
% graph_featureweight=final_ck(:,406);
run_index=[1:10];
 feature_weight = zeros(10,40);
for i=1:10
[train,text]=xlsread(['..\result\multi run\run',num2str(run_index(i)),'\10%split.xlsx'],'train dataset');
[test,text]=xlsread(['..\result\multi run\run',num2str(run_index(i)),'\10%split.xlsx'],'test dataset');
load(['..\result\multi run\run',num2str(run_index(i)),'\feature weight10%.mat'])
final_ck(:,all(final_ck==0,1)) = [];
feature_weight(i,:)=final_ck(:,end);
end

graph_featureweight = = mean(feature_weight,1);
% graph_featureweight2 = final_ck(:,406);

time_index=[1,7:14,35,36];
frequency_index=[2:6,32:34,37:40];
time_frequency_index=[15:31];
new_graphfeature_weight=[graph_featureweight(time_index)',graph_featureweight(frequency_index)',graph_featureweight(time_frequency_index)'];
shapley_value = feature_weight_class1;
shapley_value = mean(abs(shapley_value),2);
shapley_value = shapley_value/max(shapley_value);
new_shapley_value=[shapley_value(time_index);shapley_value(frequency_index);shapley_value(time_frequency_index)];
newpaper_feature_weight=[paper_feature_weight(time_index),paper_feature_weight(frequency_index),paper_feature_weight(time_frequency_index)];

[sort_graphvalue,sort_graphindex]=sort(new_graphfeature_weight,'descend');
[sort_shapleyvalue,sort_shapleyindex]=sort(new_shapley_value,'descend');
[sort_papervalue,sort_paperindex]=sort(newpaper_feature_weight,'descend');

%% 

[labelednum,~]=size(train);
[unlabelednum,~]=size(test);
disp(unlabelednum)
disp('=========start========');
class_test=test(:,1);
feature_train_test=[train(:,2:end);test(:,2:end)];
new_feature_train_test=[feature_train_test(:,time_index),feature_train_test(:,frequency_index),feature_train_test(:,time_frequency_index)];
feature_weight = new_graphfeature_weight;
%permuta features

graph_featurepermuta = zeros(6,1);
sort_index=sort_graphindex;
for i=1:6
    if i==1
        new_feature_train_test=new_feature_train_test;
    end
    
if i==2
permuta_index=sort_index(1);
randIndex = randperm(444);
permuta_features= new_feature_train_test(randIndex,permuta_index);
new_feature_train_test(:,permuta_index) = permuta_features;
end

if i==3
permuta_index1=sort_index(1);
permuta_index2=sort_index(2);
randIndex1 = randperm(444);
randIndex2 = randperm(444);
permuta_features1= new_feature_train_test(randIndex1,permuta_index1);
permuta_features2= new_feature_train_test(randIndex2,permuta_index2);
new_feature_train_test(:,permuta_index1) = permuta_features1;
new_feature_train_test(:,permuta_index2) = permuta_features2;
end

if i==4
permuta_index1=sort_index(1);
permuta_index2=sort_index(2);
permuta_index3=sort_index(3);
randIndex1 = randperm(444);
randIndex2 = randperm(444);
randIndex3 = randperm(444);
permuta_features1= new_feature_train_test(randIndex1,permuta_index1);
permuta_features2= new_feature_train_test(randIndex2,permuta_index2);
permuta_features3= new_feature_train_test(randIndex3,permuta_index3);
new_feature_train_test(:,permuta_index1) = permuta_features1;
new_feature_train_test(:,permuta_index2) = permuta_features2;
new_feature_train_test(:,permuta_index3) = permuta_features3;
end

if i==5
permuta_index1=sort_index(1);
permuta_index2=sort_index(2);
permuta_index3=sort_index(3);
permuta_index4=sort_index(4);
randIndex1 = randperm(444);
randIndex2 = randperm(444);
randIndex3 = randperm(444);
randIndex4 = randperm(444);
permuta_features1= new_feature_train_test(randIndex1,permuta_index1);
permuta_features2= new_feature_train_test(randIndex2,permuta_index2);
permuta_features3= new_feature_train_test(randIndex3,permuta_index3);
permuta_features4= new_feature_train_test(randIndex4,permuta_index4);
new_feature_train_test(:,permuta_index1) = permuta_features1;
new_feature_train_test(:,permuta_index2) = permuta_features2;
new_feature_train_test(:,permuta_index3) = permuta_features3;
new_feature_train_test(:,permuta_index4) = permuta_features4;
end

if i==6
permuta_index1=sort_index(1);
permuta_index2=sort_index(2);
permuta_index3=sort_index(3);
permuta_index4=sort_index(4);
permuta_index5=sort_index(5);
randIndex1 = randperm(444);
randIndex2 = randperm(444);
randIndex3 = randperm(444);
randIndex4 = randperm(444);
randIndex5 = randperm(444);
permuta_features1= new_feature_train_test(randIndex1,permuta_index1);
permuta_features2= new_feature_train_test(randIndex2,permuta_index2);
permuta_features3= new_feature_train_test(randIndex3,permuta_index3);
permuta_features4= new_feature_train_test(randIndex4,permuta_index4);
permuta_features5= new_feature_train_test(randIndex5,permuta_index5);
new_feature_train_test(:,permuta_index1) = permuta_features1;
new_feature_train_test(:,permuta_index2) = permuta_features2;
new_feature_train_test(:,permuta_index3) = permuta_features3;
new_feature_train_test(:,permuta_index4) = permuta_features4;
new_feature_train_test(:,permuta_index5) = permuta_features5;
end

%
train_label=train(:,1);
test_label=test(:,1);
initial_label=[train_label;zeros(unlabelednum,1)];
initial_label_index = logical([ones(labelednum,1); zeros(unlabelednum, 1)]);
class_train_test=[train_label;test_label];

x_known=[train_label;ones(length(class_train_test)-length(train_label),1)*0];


for j = 1:size(new_feature_train_test,2)

    FD{j} = (new_feature_train_test(:,j) - new_feature_train_test(:,j).').^2;
    
end

W = zeros(size(new_feature_train_test,1));

for j = 1:size(new_feature_train_test,2)
    
    W = W + feature_weight(j) * FD{j};
    
end

W = exp(-W);

W(W == diag(W)) = 0;
L = diag(sum(W))-W;


W_ele=sum(W);
        D=diag(W_ele);
        L_norm=D^(-1/2)*L*D^(-1/2);
        n = size(class_train_test,1);
        W_normalized = W./max(W(:));
        x_known=[train_label;ones(length(class_train_test)-length(train_label),1)*0];
        [nRows,nCols] = size(W_normalized);
        C_matrix=zeros(nRows,nCols);
        subsetIdx = 1:length(train_label);
        diagonalIdx = (subsetIdx-1) * (nRows + 1) + 1;
        C_matrix(diagonalIdx) = 1;
        T=isnan(L_norm);
        [row_L_norm,~]=size(L_norm);
    
        if sum(sum(T))~=0
            L_norm=ones(row_L_norm);
        end
        
    cvx_begin sdp
        variable x(n,1);
        variable X(n,n) symmetric;
        minimize(trace(L_norm*X))
        subject to
        diag(X) == 1;
        [X x; x' 1] >= 0;
        %% ======FIX THE KNOWN LABELS=====
        x(initial_label_index) == class_train_test(initial_label_index);
        %% ===============================
    cvx_end
     
    final_label=sign(x(44:end));
    accuracy=1-length(find(final_label~=test_label))/length(x(44:end));
    graph_featurepermuta(i)=accuracy;
end
%     save('E:\PHD\SPM paper\graph feature learning\Louisiana_microearthquake\result\upper100,ckinitial 1, no normal\feature permute\graph_feature_permuta5.mat','graph_featurepermuta')
 %%
 
 clear
 clc
 close all
 load('graph_feature_permuta.mat');
 load('shapley_feature_permuta.mat');
 load('paper_feature_permuta.mat');
 
graph1=graph_featurepermuta(2)-graph_featurepermuta(1);
graph2=graph_featurepermuta(3)-graph_featurepermuta(1);
graph3=graph_featurepermuta(4)-graph_featurepermuta(1);
graph4=graph_featurepermuta(5)-graph_featurepermuta(1);
graph5=graph_featurepermuta(6)-graph_featurepermuta(1);

graph_value=[-graph1,-graph2,-graph3,-graph4,-graph5];


shapley1=shapley_featurepermuta(2)-shapley_featurepermuta(1);
shapley2=shapley_featurepermuta(3)-shapley_featurepermuta(1);
shapley3=shapley_featurepermuta(4)-shapley_featurepermuta(1);
shapley4=shapley_featurepermuta(5)-shapley_featurepermuta(1);
shapley5=shapley_featurepermuta(6)-shapley_featurepermuta(1);

shapley_value=[-shapley1,-shapley2,-shapley3,-shapley4,-shapley5];

paper1=paper_featurepermuta(2)-paper_featurepermuta(1);
paper2=paper_featurepermuta(3)-paper_featurepermuta(1);
paper3=paper_featurepermuta(4)-paper_featurepermuta(1);
paper4=paper_featurepermuta(5)-paper_featurepermuta(1);
paper5=paper_featurepermuta(6)-paper_featurepermuta(1);

paper_value=[-paper1,-paper2,-paper3,-paper4,-paper5];


stem(graph_value)
hold on
stem(shapley_value)
hold on
stem(paper_value)
 
 
 
 