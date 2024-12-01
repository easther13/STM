function choose=Divide(label,k,randomSeed)

%% Divide data into k group based on k-fold cross validation
%rand('state',randomSeed);
choose=cell(k,1);
labelmid=label;
c=ones(1,length(labelmid));                                                    % Count data number
sy=cumsum(c);                                                                  % Assign sequence number

%% k-fold cross validation
for s=k:-1:2    
    c=ones(1,length(labelmid));                                                    % Count data number 
    c=cumsum(c);                                                                   % Assign sequence number
    a=unique(labelmid);                                                          %make sure the class of the data
    Lc=length(a);                                                                  % Count class number
    all=0;
    for i=1:Lc
        Ad=find(labelmid==a(i));
        Ai=Ad;
        for j=1:length(Ad)*1/s     
            t=ceil(rand(1)*length(Ai));%生成一个随机整数 t，用于选择 Ai 中的一个样本。
  %         fprintf('Rand value is %g\n',rand('seed'));
            if t>length(Ai)%确保t不会超过Ai的长度，以防止数组越界
                t=length(Ai);
            end
            all=all+1;
            boostplace(all)=Ai(t);%确保 t 不会超过 Ai 的长度，以防止数组越界
            Ai=setdiff(Ai,Ai(t));%从 Ai 中移除已经选择的样本，确保不会重复选择同一个样本
        end
    end
    choose{s,1}=sy(1,boostplace);%从集合 sy 中选择索引为 boostplace 的元素，并将这些元素组成一个列向量;每个元素 choose{s,1} 是一个列向量
    a=setdiff(c,boostplace);
    labelmid=labelmid(a,1);
    sy=sy(setdiff(c,boostplace));
end
choose{1,1}=sy;
end