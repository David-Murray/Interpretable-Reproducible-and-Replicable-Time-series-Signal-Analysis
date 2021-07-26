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
time_index=[1,7:14,35,36];
frequency_index=[2:6,32:34,37:40];
time_frequency_index=[15:31];

ave_feature_weight = ave_feature_weight/max(ave_feature_weight);


% figure
new_ave_feature_weight=[ave_feature_weight(time_index),ave_feature_weight(frequency_index),ave_feature_weight(time_frequency_index)];
stem(new_ave_feature_weight)

paper_feature_weight=[0.080,0.445,0.286,0.278,0.27,0.092,0.0501,0.358,0.0901,0.052,0.261,0.205,0.254,0.076,0.167,0.262,0.122,0.101,0.094,0,0.081,0,0,0,0,0.088,0.214,0,0,0.21,0,0.13,0.132,0.050,0.245,0.234,0.127,0.111,0,0];
paper_feature_weight = paper_feature_weight/max(paper_feature_weight);
newpaper_feature_weight=[paper_feature_weight(time_index),paper_feature_weight(frequency_index),paper_feature_weight(time_frequency_index)];

hold on
stem(newpaper_feature_weight)

%%
time_index=[1,7:14,35,36];
frequency_index=[2:6,32:34,37:40];
time_frequency_index=[15:31];

load('shapley_value_class1.mat')
load('shapley_value_class2.mat')
shapley_value = feature_weight_class1;
shapley_value = mean(abs(shapley_value),2);
shapley_value = shapley_value/max(shapley_value);
new_shapley_value=[shapley_value(time_index);shapley_value(frequency_index);shapley_value(time_frequency_index)];

hold on
stem(new_shapley_value)
%%

%% time feature
text(1,0,'maxAmp','rotation',90,'FontName','Times New Roman','FontSize',30)
text(2,0,'Energy','rotation',90,'FontName','Times New Roman','FontSize',30)
text(3,0,'rect','rotation',90,'FontName','Times New Roman','FontSize',30)
text(4,0,'azmth','rotation',90,'FontName','Times New Roman','FontSize',30)
text(5,0,'indAngl','rotation',90,'FontName','Times New Roman','FontSize',30)
text(6,0,'Dip','rotation',90,'FontName','Times New Roman','FontSize',30)
text(7,0,'DipRec','rotation',90,'FontName','Times New Roman','FontSize',30)
text(8,0,'ccD','rotation',90,'FontName','Times New Roman','FontSize',30)
text(9,0,'ccS','rotation',90,'FontName','Times New Roman','FontSize',30)
text(10,0,'envelopD','rotation',90,'FontName','Times New Roman','FontSize',30)
text(11,0,'envelopS','rotation',90,'FontName','Times New Roman','FontSize',30)

%% frequency feature

text(12,0,'spCen','rotation',90,'FontName','Times New Roman','FontSize',30)
text(13,0,'rmsA','rotation',90,'FontName','Times New Roman','FontSize',30)
text(14,0,'maxPFA','rotation',90,'FontName','Times New Roman','FontSize',30)
text(15,0,'maxFA','rotation',90,'FontName','Times New Roman','FontSize',30)
text(16,0,'maxPF_FA','rotation',90,'FontName','Times New Roman','FontSize',30)
text(17,0,'skwnss','rotation',90,'FontName','Times New Roman','FontSize',30)
text(18,0,'semblanceD','rotation',90,'FontName','Times New Roman','FontSize',30)
text(19,0,'semblanceS','rotation',90,'FontName','Times New Roman','FontSize',30)
text(20,0,'udD','rotation',90,'FontName','Times New Roman','FontSize',30)
text(21,0,'udS','rotation',90,'FontName','Times New Roman','FontSize',30)
text(22,0,'ud2D','rotation',90,'FontName','Times New Roman','FontSize',30)
text(23,0,'ud2S','rotation',90,'FontName','Times New Roman','FontSize',30)

%% time_frequency feature

text(24,0,'xotsu','rotation',90,'FontName','Times New Roman','FontSize',30)
text(25,0,'ccnAb2D','rotation',90,'FontName','Times New Roman','FontSize',30)
text(26,0,'ccnAb2S','rotation',90,'FontName','Times New Roman','FontSize',30)
text(27,0,'ypicD','rotation',90,'FontName','Times New Roman','FontSize',30)
text(28,0,'ypicS','rotation',90,'FontName','Times New Roman','FontSize',30)
text(29,0,'ccnRel2D','rotation',90,'FontName','Times New Roman','FontSize',30)
text(30,0,'ccnRel2S','rotation',90,'FontName','Times New Roman','FontSize',30)
text(31,0,'ccAb2D','rotation',90,'FontName','Times New Roman','FontSize',30)
text(32,0,'ccAb2S','rotation',90,'FontName','Times New Roman','FontSize',30)
text(33,0,'ccRel2D','rotation',90,'FontName','Times New Roman','FontSize',30)
text(34,0,'ccRel2S','rotation',90,'FontName','Times New Roman','FontSize',30)
text(35,0,'xp2D','rotation',90,'FontName','Times New Roman','FontSize',30)
text(36,0,'xp2S','rotation',90,'FontName','Times New Roman','FontSize',30)
text(37,0,'xpp2D','rotation',90,'FontName','Times New Roman','FontSize',30)
text(38,0,'xpp2S','rotation',90,'FontName','Times New Roman','FontSize',30)
text(39,0,'semD','rotation',90,'FontName','Times New Roman','FontSize',30)
text(40,0,'semS','rotation',90,'FontName','Times New Roman','FontSize',30)
%% backup
% text(1,0,'maxAmp','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(2,0,'spCen','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(3,0,'rmsA','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(4,0,'maxPFA','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(5,0,'maxFA','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(6,0,'maxPF_FA','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(7,0,'Energy','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(8,0,'rect','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(9,0,'azmth','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(10,0,'indAngl','rotation',90,'FontName','Times New Roman','FontSize',36)
% 
% 
% text(11,0,'Dip','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(12,0,'DipRec','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(13,0,'ccD','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(14,0,'ccS','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(15,0,'xotsu','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(16,0,'ccnAb2D','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(17,0,'ccnAb2S','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(18,0,'ypicD','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(19,0,'ypicS','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(20,0,'ccnRel2D','rotation',90,'FontName','Times New Roman','FontSize',36)
% 
% text(21,0,'ccnRel2S','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(22,0,'ccAb2D','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(23,0,'ccAb2S','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(24,0,'ccRel2D','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(25,0,'ccRel2S','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(26,0,'xp2D','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(27,0,'xp2S','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(28,0,'xpp2D','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(29,0,'xpp2S','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(30,0,'semD','rotation',90,'FontName','Times New Roman','FontSize',36)
% 
% text(31,0,'semS','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(32,0,'skwnss','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(33,0,'semblanceD','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(34,0,'semblanceS','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(35,0,'envelopD','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(36,0,'envelopS','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(37,0,'udD','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(38,0,'udS','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(39,0,'ud2D','rotation',90,'FontName','Times New Roman','FontSize',36)
% text(40,0,'ud2S','rotation',90,'FontName','Times New Roman','FontSize',36)



