function model = fun_solver_GAP_DVI(X,Y,eta,Cp, Cp_pre, w_pre,alpha)
%% 初始化
iter = 1;m=length(Y);
ind_R = (1:length(Y))';%指标排序
PG=zeros(length(Y),1);
Q = 1/eta*X*X';
w = 1/eta*X'*(alpha.*Y);
dual_gap_value = compute_gap(X, Y, w, alpha, eta, Cp);
% screen_ratio(iter) = 0;
screen_ratio_dvi=0;
% screen_ratio_gap=0;
t_scr_dvi = 0;
t_scr_gap = 0;
t_dcdm = 0;
%% DVI筛选准则
if dual_gap_value>1e-4
    tt = tic;
    Mi = (Cp+Cp_pre)/(2*Cp_pre)*Y(ind_R).*X(ind_R,:)*w_pre-abs(Cp-Cp_pre)/(2*Cp_pre)*sqrt(sum(X(ind_R,:).^2,2))*norm(w_pre);
    Ma = (Cp+Cp_pre)/(2*Cp_pre)*Y(ind_R).*X(ind_R,:)*w_pre+abs(Cp-Cp_pre)/(2*Cp_pre)*sqrt(sum(X(ind_R,:).^2,2))*norm(w_pre);
    ind_0_d = find(Mi>1);
    ind_C_d = find(Ma<1);
    screen_ratio_dvi = length([ind_0_d;ind_C_d]);
    alpha(ind_R(ind_0_d)) = 0;
    alpha(ind_R(ind_C_d)) = Cp;
    ind_R([ind_0_d; ind_C_d]) = [];
    t_scr_dvi = toc(tt);
    if ~isempty([ind_0_d; ind_C_d])
        w = 1/eta*sum(alpha.*Y.*X,1)';%更新w
    end
    dual_gap_value = compute_gap(X, Y, w, alpha, eta, Cp);
end
DG(iter) = dual_gap_value;
%% 开始迭代
while dual_gap_value>1e-6 && iter<200
% while dual_gap_value>1e-4 && iter<200
    if(mod(iter,10)==0)
%     if(mod(iter,30)==0)
        % Gap筛选准则
        t1 = tic;
        Mi = Y(ind_R).*X(ind_R,:)*w - sqrt(sum(X(ind_R,:).^2,2))*sqrt(dual_gap_value);
        Ma = Y(ind_R).*X(ind_R,:)*w + sqrt(sum(X(ind_R,:).^2,2))*sqrt(dual_gap_value);
        ind_0 = find(Mi>1);
        ind_C = find(Ma<1);
        alpha(ind_R(ind_0)) = 0;
        alpha(ind_R(ind_C)) = Cp;
        ind_R([ind_0; ind_C]) = [];
        t_scr_gap = t_scr_gap+toc(t1);
        if ~isempty([ind_0; ind_C])
            w = 1/eta*sum(alpha.*Y.*X,1)';
        end
    end
    % 求解约简的子问题
    t2 = tic;
    for i = 1 : length(ind_R)
        alpha_ = alpha;
        deta_G = Y(ind_R(i))*w'*X(ind_R(i),:)'-1;%同G
        if abs(alpha(ind_R(i))) < 1e-6
            PG(ind_R(i)) = min(deta_G, 0);
        elseif abs(alpha(ind_R(i))-Cp)<1e-6
            PG(ind_R(i)) = max(deta_G, 0);
        else
            PG(ind_R(i)) = deta_G;
        end
        if abs(PG(ind_R(i)))>1e-6
            alpha(ind_R(i)) = min(max(alpha(ind_R(i))-deta_G/Q(ind_R(i),ind_R(i)), 0), Cp);
            w = w+(alpha(ind_R(i))-alpha_(ind_R(i)))*1/eta*Y(ind_R(i))*X(ind_R(i),:)';%w等于1/eta*alpha*Y*X
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
model.t_scr_dvi = t_scr_dvi;
model.t_scr_gap = t_scr_gap;
model.t_dcdm= t_dcdm;
model.screen_ratio_dvi = screen_ratio_dvi;
model.screen_ratio_gap = m-length(ind_R)-screen_ratio_dvi;
model.ind_R=ind_R;
model.time=t_scr_dvi+t_scr_gap+ t_dcdm;
realnumber=sum(alpha==0)+sum(alpha==Cp);
model.realnumber=realnumber;
model.ratio=(m-length(ind_R))/realnumber;
end
function dual_gap_value = compute_gap(X, Y, w, alpha, eta, Cp)
primal_obj_value = 0.5*eta*norm(w)^2+Cp*sum(max(0,1-Y.*X*w));
dual_obj_value = -0.5*(1/eta)*alpha'*diag(Y)*X*X'*diag(Y)*alpha+ones(length(Y),1)'*alpha;
dual_gap_value = primal_obj_value-dual_obj_value;
end