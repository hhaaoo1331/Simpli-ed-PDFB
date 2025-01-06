function [Diff] = Second_differencematrix(m,n)

Dx1 = spdiags(-ones(m,1),0,m,m);
Dx2 = spdiags(ones(m,1),1,m,m);
Dx = Dx1 + Dx2;
Dx(m,:) = 0;


Dy1 = spdiags(-ones(n,1),0,n,n);
Dy2 = spdiags(ones(n,1),1,n,n);
Dy = Dy1 + Dy2;
Dy(n,:) = 0;


Diff = [kron(speye(n,n),-Dx'*Dx);kron(-Dy'*Dy,speye(m,m))];

% Diff = [kron(D,speye(m,n));kron(speye(m,n),D)];
% Diff = [kron(D,speye(m,n)); kron(speye(m,n),D)];