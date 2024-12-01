function model = DCDM(X,Y,X_C,Y_C,C,f,w_pre,object_pre,tol_num, alpha_int)
%% this is the dual coordinate descent algorithm for linear-CSVM;
% min 1/2*a'*Y*X*X'*Y*a + f'*a
% s.t. 0<=a<=C
%% Initialization
[m,n] = size(X);
alpha = alpha_int; %% alpha is the initial value given advance;
w = X'*(alpha.*Y);
object = 0.5*w'*w +f'*alpha; 
Q = sum(X.^2,2);
for k=1:tol_num 
    order = randperm(m);
    M_max = 0;
    M_min = 0;
    for i = 1:m
        j = order(i);
        G = Y(j)*X(j,:)*w+f(j);
        if alpha(j)==0
            PG = min(G,0);
        elseif alpha(j)==C
            PG = max(G,0);
        else
            PG = G;
        end
        M_max = max(M_max,PG);
        M_min = min(M_min,PG);
        if PG~=0
            alpha_old = alpha(j);
            alpha(j) = min(max(alpha(j)-G/Q(j),0),C);
            w = w + (alpha(j)-alpha_old)*X(j,:)'*Y(j); %%update w;
        end
    end
    k=k+1;
    w_all = w+w_pre;
    s1 =[Y;Y_C].*([X;X_C]*w_all);%每个数据点的函数值
    object = 0.5*w'*w + f'*alpha +object_pre;
    P_object = 0.5*w_all'*w_all + C*sum(max(0,1-s1));
    Gap = P_object + object;
    %     Gap
      Stop = M_max - M_min;
    %   plot(k, Gap, 'r*-', k, P_object, 'b^')
    %   hold on
    if Gap<= 1e-3  || k==tol_num || Stop<1e-6
        break;
    end
end
model.w = w +w_pre;
model.x = alpha;
model.object = object;
end
