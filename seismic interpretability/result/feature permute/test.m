%%
clear
clc

[data,text]=xlsread('feature permute result.xlsx');
graph_featurepermuta = data(6,2:end);
shapley_featurepermuta = data(15,2:end);
paper_featurepermuta = data(24,2:end);

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


