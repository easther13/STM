function model = DCDM1(X,Y,eta,Cp,alpha_int)
%% 初始化
iter = 1;
ind_R = (1:length(Y))';
PG=zeros(length(Y),1);
Q = 1/eta*X*X';
alpha=alpha_int;
w = 1/eta*X'*(alpha.*Y);
dual_gap_value = compute_gap(X, Y, w, alpha, eta, Cp);
t_dcdm = 0;
DG(iter) = dual_gap_value;
%% 开始迭代
while dual_gap_value>1e-4 && iter<200
    t2 = tic;
    for i = 1 : length(ind_R)
        alpha_ = alpha;
        deta_G = Y(ind_R(i))*w'*X(ind_R(i),:)'-1;
        if abs(alpha(ind_R(i))) < 1e-6
            PG(ind_R(i)) = min(deta_G, 0);
        elseif abs(alpha(ind_R(i))-Cp)<1e-6
            PG(ind_R(i)) = max(deta_G, 0);
        else
            PG(ind_R(i)) = deta_G;
        end
        if abs(PG(ind_R(i)))>1e-6
            alpha(ind_R(i)) = min(max(alpha(ind_R(i))-deta_G/Q(ind_R(i),ind_R(i)), 0), Cp);
            w = w+(alpha(ind_R(i))-alpha_(ind_R(i)))*1/eta*Y(ind_R(i))*X(ind_R(i),:)';
        end
    end
    t_dcdm = t_dcdm+toc(t2);
    dual_gap_value = compute_gap(X, Y, w, alpha, eta, Cp);
    iter = iter + 1;
    DG(iter) = dual_gap_value;
end

model.w = w;
model.x = alpha;
model.DG = DG; 
model.t_dcdm= t_dcdm;
end
function dual_gap_value = compute_gap(X, Y, w, alpha, eta, Cp)
primal_obj_value = 0.5*eta*norm(w)^2+Cp*sum(max(0,1-Y.*X*w));
dual_obj_value = -0.5*(1/eta)*alpha'*diag(Y)*X*X'*diag(Y)*alpha+ones(length(Y),1)'*alpha;
dual_gap_value = primal_obj_value-dual_obj_value;
end