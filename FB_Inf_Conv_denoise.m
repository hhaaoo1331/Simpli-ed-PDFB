function [x_update,k,SNR,SSIM,PSNR,t2]= FB_Inf_Conv_denoise(f_true,f,iter,epsilon,L,K,M,lambda1,lambda2,theta1,theta2,gamma1,gamma2,tau,sigma,a1)

%
% This function solves the ROF model
% \min_{x} 1/2 || x -f ||^2 + lambda* ||x||_{TV} 
%
%
% Input 
% f_true ----------- 
% f ---------------- the noisy image
% y0 ---------------
% lambda ----------- the regularization parameter
% gamma ------------
% iter -------------
% epsilon ----------
% a1 --------------- the relaxation parameter

 [m,n]=size(f);
% y =zeros(m,2*n);

fn = f(:);

x = zeros(m*n,1);
x = fn;
v = zeros(size(L,1),1);



[mk,nk] = size(K);
z = zeros(nk,1);
p = zeros(mk,1);



[mm,nm] = size(M);
y = zeros(nm,1);
q = zeros(mm,1);


% theta1 = 0.1;
% gamma1 = 0.1;
% theta2 = 0.2;
% gamma2 = 0.2;
% a1 = 1;
% tau = 1;
% sigma = 0.4;


k =1;

SNR = [];


 tic;
t1 = clock;

while k <= iter
    

    xbar = proj_bound(x - tau*( (x-fn) + L'*v  ),0,255);
    
    
    p1 = (1/theta1)*p + K*z;
    pbar =  theta1*( p1 - max(abs(p1)-lambda1/theta1,0).*sign(p1) );
    
     q1 = (1/theta2)*q + M*y;
    qbar =  theta2*( q1 - max(abs(q1)-lambda2/theta2,0).*sign(q1) );
    
    u1 = z + gamma1*( K'*(p-2*pbar) + v + sigma*L*(2*xbar-x)  );
    u2 = y + gamma2*( M'*(q-2*qbar) + v + sigma*L*(2*xbar-x)  );
    
    zbar = (1+sigma*gamma2)/(1+sigma*(gamma1+gamma2))*( u1 - (sigma*gamma1)/(1+sigma*gamma2)*(u2) );
    ybar = 1/(1+sigma*gamma2) * (u2 - sigma*gamma2*zbar);
    vbar = v + sigma*( L*(2*xbar - x) - zbar - ybar );
    
    x_update = (1-a1)*x + a1 * xbar;
    
    p_update = (1-a1)*p + a1 * pbar;
    q_update = (1-a1)*q + a1 * qbar;
    z_update = (1-a1)*z + a1 * zbar;
    y_update = (1-a1)*y + a1 * ybar;
    v_update = (1-a1)*v + a1 * vbar;
    
      % isotropic total variation
%       y1_sum = sqrt(y1(1:m*n).^2 + y1(m*n+1:end).^2);
%       y11= max(y1_sum - lambda/gamma,0).*(y1(1:m*n)./y1_sum);
%       y12 = max(y1_sum - lambda/gamma,0).*(y1(m*n+1:end)./y1_sum);
%       y_update = (1-a1)*y + a1 * (y1-[y11;y12]);


      % isotropic total variation
%       y1_sum = sqrt(y1(:,1:n).^2 + y1(:,n+1:end).^2);
%       y11= max(y1_sum - lambda/gamma,0).*(y1(:,1:n)./y1_sum);
%       y12 = max(y1_sum - lambda/gamma,0).*(y1(:,n+1:end)./y1_sum);
%       y_update = (1-a1)*y + a1 * (y1-[y11,y12]);
      
      
      
   %  y_update = (1-a1)*y1 + a1 *( y1 - max(abs(y1)-lambda/gamma,0).*sign(y1));   %
   
        SNR(k)   = 20*log10(norm(f_true(:))/norm(f_true(:)-x_update(:)));
        SSIM = ssim(f_true/255,reshape(x_update,m,n)/255);
       PSNR(k)  = 20*log10(sqrt(m*n)*255/norm(f_true(:)-x_update(:))) ;
   %     fval(k) = 0.5*norm(x_update-f)^2 + lambda*norm(D*x_update,1);
    
        t2(k)=etime(clock,t1);   
   
    if  norm(x_update(:)-x(:))/norm(x(:)) <= epsilon
        break;
    else
       p = p_update;
       q = q_update;
       z = z_update;
       y = y_update; 
       v = v_update;
       x = x_update;
       k = k+1;
    end
    
 
% u=f+lamda*div(y1_update,y2_update,1);

% mse(k) = norm(f_true - u,'fro')^2/(m*n);

end
end

