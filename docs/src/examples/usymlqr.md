```@example usymlqr
using Krylov, LinearOperators, LDLFactorizations
using LinearAlgebra, Printf, SparseArrays

# Identity matrix.
eye(n::Int) = sparse(1.0 * I, n, n)

# Saddle-point systems
n = m = 5
A = [2^(i/j)*j + (-1)^(i-j) * n*(i-1) for i = 1:n, j = 1:n]
b = ones(n)
D = diagm(0 => [2.0 * i for i = 1:n])
m, n = size(A)
c = -b

# [D   A] [x] = [b]
# [Aᴴ  0] [y]   [c]
llt_D = cholesky(D)
opD⁻¹ = LinearOperator(Float64, 5, 5, true, true, (y, v) -> ldiv!(y, llt_D, v))
opH⁻¹ = BlockDiagonalOperator(opD⁻¹, eye(n))
(x, y, stats) = usymlqr(A, b, c, M=opD⁻¹, sp=true)
K = [D A; A' zeros(n,n)]
B = [b; c]
r = B - K * [x; y]
resid = sqrt(dot(r, opH⁻¹ * r))
@printf("USYMLQR: Relative residual: %8.1e\n", resid)
```
