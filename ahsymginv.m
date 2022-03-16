% Given an  m-by-n matrix M and r <= rank(M),
% [H,A]=ahsymginv(M,r) returns a rank-r approximation A of M, and an n-by-m
% matrix H satisfying (i) AHA=A, (ii) HAH=H, (iii) (AH)'=AH, (iv) having only r
% nonzero rows, and with low (vector) 1-norm. Such an H is a particular 
% sparse, block-structured, minimum-rank generalized inverse of A, for which 
% x=Hb also solves the least-squares problem min ||Ax-b||_2. If r=rank(M), then A=M.
%
% [H,A,K]=ahsymginv(M,r), with three outputs arguments, also returns K, a list of 
% r elements from 1:n, so that the dense r-by-m submatrix H_hat of H is given by 
% pinv(A(:,K)). K is computed by a local-search algorithm, which aims to construct 
% a lower (vector) 1-norm  matrix H_hat.
%
% ahsymginv(M,r) uses [R,C]=nsub(A,r) to initialize its local-search algorithm.
%
% [H,A,K]=ahsymginv(M,r,R,C) functions as described above, but uses (R,C) to initialize 
% its local-search algorithm, where R should be a list of r elements from 1:m, 
% C should be a list of r elements from 1:n, and A(R,C) should be nonsingular.
% 
% See: M. Fampa, J. Lee, G. Ponte, L. Xu. Experimental analysis of 
% local search for sparse reflexive generalized inverses. Journal of 
% Global Optimization, 81:1057-1093, 2021. 
%
% Example:
% 
% r = 3; 
% M = [-1 -1 1 1 -5; -1 -1 1 0 -5; 0 0 0 1 0; 2 1 1 1 1];
% 
% [H,A,K] = ahsymginv(M,r);
% 
% Result:
%   H =  [[-0.1481    0.2593   -0.4074    0.5556];
%         [ 0         0         0         0];
%         [ 0         0         0         0];
%         [ 0.3333   -0.3333    0.6667    0.0000];
%         [-0.0370   -0.1852    0.1481   -0.1111]]   
% 
%   rank(H) = 3;
%   A=M;
% Note that, selecting for example b=[1;1;1;1], we have
% norm(A*H*b-b,2)=norm(A*pinv(A)*b-b,2)=0.5774

function [H,A,K] = ahsymginv(M,r,R,C)
rM=rank(M,10^(-6));
[m,n] = size(M);
if rM==r
    A=M;
elseif rM>r
    [U,S,V]=svd(M);
    A=U(:,1:r)*S(1:r,1:r)*V(:,1:r)';
else
    error(['rank(M) = ', num2str(rM),'. Please set r <= rank(M).']);
end
if ~exist('R','var') || ~exist('C','var')
     % third parameter does not exist, so default it to something
      [R,C] = nsub(A,r);
else
    % check R and C
    if length(R)==r && length(C)==r && all(ismember(R, (1:m))) && all(ismember(C, (1:n))) 
      if (rank(A(R,C),10^(-6)) < r)
          warning('The submatrix (of the rank-r approximation of M) indexed by (R,C) is singular. Initializing instead using nsub.');
          [R,C] = nsub(A,r);
      end
    else
      warning('The index pair (R,C) is invalid. Initializing instead using nsub.');
      [R,C] = nsub(A,r);
    end
end
swaps = 0;
%  Current rows = R and current columns = C
Cb = [];
for j = (1:n)    
    n1 = find(C==j);    
    if isempty(n1)==1        
        Cb = [Cb;j];
    end
end
Ar = A(R,C);  % Initial block for local search
flag = 1;
% Local Search for C
while flag> 0    
    Ar = A(R,C);    
    flag = 0;    
    [L2,U2,P2] = lu(Ar);    
    for i = (1: n-r)        
        b = P2 * A(R,Cb(i));        
        y = L2\b;        
        alfa = U2\y;        
        % Changing C       
        [biggest_alfa,local_alfa] = max(abs(alfa));        
        if abs(biggest_alfa) > 1            
            swaps = swaps + 1;            
            el_save = C(local_alfa) ;            
            C(local_alfa) = Cb(i);            
            Cb(i) = el_save;            
            Ar = A(R,C);            
            [L2,U2,P2] = lu(Ar);            
            flag = flag + 1;
        end        
    end   
end
A_hat =  A(:,C);
H_hat = (A_hat'*A_hat) \ A_hat';
H = zeros(n,m);
for i = (1:r)
    H(C(i),:) = H_hat(i,:);
end
K=sort(C);
end
