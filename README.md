Given an  m-by-n matrix M and r <= rank(M), 
[H,A]=ahsymginv(M,r) returns a rank-r approximation A of M, and an n-by-m
matrix H satisfying (i) AHA=A, (ii) HAH=H, (iii) (AH)'=AH, (iv) having only r
nonzero rows, and with low (vector) 1-norm. Such an H is a particular 
sparse, block-structured, minimum-rank generalized inverse of A, for which 
x=Hb also solves the least-squares problem min ||Ax-b||_2. If r=rank(M), then A=M.

[H,A,K]=ahsymginv(M,r), with three outputs arguments, also returns K, a list of 
r elements from 1:n, so that the dense r-by-m submatrix H_hat of H is given by 
pinv(A(:,K)). K is computed by a local-search algorithm, which aims to construct 
a lower (vector) 1-norm  matrix H_hat.

ahsymginv(M,r) uses [R,C]=nsub(A,r) to initialize its local-search algorithm.

[H,A,K]=ahsymginv(M,r,R,C) functions as described above, but uses (R,C) to initialize 
its local-search algorithm, where R should be a list of r elements from 1:m, 
C should be a list of r elements from 1:n, and A(R,C) should be nonsingular.

See: M. Fampa, J. Lee, G. Ponte, L. Xu. Experimental analysis of 
local search for sparse reflexive generalized inverses. Journal of 
Global Optimization, 81:1057-1093, 2021. 

Example:

r = 3; 
M = [-1 -1 1 1 -5; -1 -1 1 0 -5; 0 0 0 1 0; 2 1 1 1 1];

[H,A,K] = ahsymginv(M,r);

Result:
  H =  [[-0.1481    0.2593   -0.4074    0.5556];
        [ 0         0         0         0];
        [ 0         0         0         0];
        [ 0.3333   -0.3333    0.6667    0.0000];
        [-0.0370   -0.1852    0.1481   -0.1111]]   

  rank(H) = 3;
  A=M;
Note that, selecting for example b=[1;1;1;1], we have
norm(A*H*b-b,2)=norm(A*pinv(A)-b,2)=0.5774
