function [Diff,L] = differencematrix(m,n)

Dx1 = spdiags(-ones(m,1),0,m,m);
Dx2 = spdiags(ones(m,1),1,m,m);
Dx = Dx1 + Dx2;
Dx(m,:) = 0;


Dy1 = spdiags(-ones(n,1),0,n,n);
Dy2 = spdiags(ones(n,1),1,n,n);
Dy = Dy1 + Dy2;
Dy(n,:) = 0;


Diff = [kron(speye(n,n),Dx);kron(Dy,speye(m,m))];

Dxx = kron(speye(n,n),Dx);
Dyy = kron(Dy,speye(m,m));

L = [-Dxx' sparse(m*n,m*n); sparse(m*n,m*n) -Dyy'];

% Diff = [kron(D,speye(m,n));kron(speye(m,n),D)];
% Diff = [kron(D,speye(m,n)); kron(speye(m,n),D)];