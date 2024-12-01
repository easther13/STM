%% Machine Learning Online Class
% Support Vector Machines

%% Initialization
% clear ; close all; clc
%% =============== Part 1: Loading and Visualizing Data ================
%  We start the exercise by first loading and visualizing the dataset.
%  The following code will load the dataset into your environment and plot
%  the data.
% fprintf('Loading and Visualizing Data ...\n')
% [cfilename]=uigetfile({'*.mat','All files(*.*)'},...
%     'Select an mat file to caculate');
% dataname  = 'Data';STRC1='CSVM_'
% STRA='C:\Users\pxl\Desktop\TSS-CSVM\TSS_CSVM_code-201712\';
% STRB1=regexp(cfilename, '.mat', 'split');
% STRB2=regexp(cfilename, '.txt', 'split');
% name =[STRA STRB1{1}];
% outname1=[STRC1 STRB2{1}];

load('HSI_Salines.mat');

global m n X_test Y_test

Y=label;
Y(Y==0) = -1;
Y=full(Y);
% Define the K-fold cross validation
K = 5;

% for s=1:40
% =============== Part 2: training data ================
X_pos = X(Y==1,:);
X_neg = X(Y==-1,:);
% idx=idxs(s);
[idx] = cross_validation(X_pos,X_neg,K);

 
%%Rank-1 STM
%     [Object.rank1, Acc.rank1,~,time.rank1] = path_quad_STM(idx);

%%HR-STM
    [Object.HR, Acc.HR,~,time.HR]= path_HRSTM(idx);

