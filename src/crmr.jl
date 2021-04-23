# An implementation of CRMR for the solution of the
# (under/over-determined or square) linear system
#
#  Ax = b.
#
# The method seeks to solve the minimum-norm problem
#
#  min ‖x‖²  s.t. Ax = b,
#
# and is equivalent to applying the conjugate residual method
# to the linear system
#
#  AAᵀy = b.
#
# This method is equivalent to Craig-MR, described in
#
# D. Orban and M. Arioli. Iterative Solution of Symmetric Quasi-Definite Linear Systems,
# Volume 3 of Spotlights. SIAM, Philadelphia, PA, 2017.
#
# D. Orban, The Projected Golub-Kahan Process for Constrained
# Linear Least-Squares Problems. Cahier du GERAD G-2014-15,
# GERAD, Montreal QC, Canada, 2014.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Montreal, QC, April 2015.

export crmr, crmr!


"""
    (x, stats) = crmr(A, b::AbstractVector{T};
                      M=opEye(), λ::T=zero(T), atol::T=√eps(T),
                      rtol::T=√eps(T), itmax::Int=0, verbose::Int=0, history::Bool=false) where T <: AbstractFloat

Solve the consistent linear system

    Ax + √λs = b

using the Conjugate Residual (CR) method, where λ ≥ 0 is a regularization
parameter. This method is equivalent to applying CR to the normal equations
of the second kind

    (AAᵀ + λI) y = b

but is more stable. When λ = 0, this method solves the minimum-norm problem

    min ‖x‖₂  s.t.  x ∈ argmin ‖Ax - b‖₂.

When λ > 0, this method solves the problem

    min ‖(x,s)‖₂  s.t. Ax + √λs = b.

CGMR produces monotonic residuals ‖r‖₂.
It is formally equivalent to CRAIG-MR, though can be slightly less accurate,
but simpler to implement. Only the x-part of the solution is returned.

A preconditioner M may be provided in the form of a linear operator.

#### References

* D. Orban and M. Arioli, *Iterative Solution of Symmetric Quasi-Definite Linear Systems*, Volume 3 of Spotlights. SIAM, Philadelphia, PA, 2017.
* D. Orban, *The Projected Golub-Kahan Process for Constrained Linear Least-Squares Problems*. Cahier du GERAD G-2014-15, 2014.
"""
function crmr(A, b :: AbstractVector{T}; kwargs...) where T <: AbstractFloat
  solver = CrmrSolver(A, b)
  crmr!(solver, A, b; kwargs...)
end

function crmr!(solver :: CrmrSolver{T,S}, A, b :: AbstractVector{T};
               M=opEye(), λ :: T=zero(T), atol :: T=√eps(T),
               rtol :: T=√eps(T), itmax :: Int=0, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, S <: DenseVector{T}}

  m, n = size(A)
  size(b, 1) == m || error("Inconsistent problem size")
  (verbose > 0) && @info @sprintf("CRMR: system of %d equations in %d variables\n", m, n)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  isa(M, opEye) || (eltype(M) == T) || error("eltype(M) ≠ $T")

  # Compute the adjoint of A
  Aᵀ = A'

  # Set up workspace.
  x, p, r = solver.x, solver.p, solver.r

  x .= zero(T)  # initial estimation x = 0
  r .= M * b    # initial residual r = M * (b - Ax) = M * b
  bNorm = @knrm2(m, r)  # norm(b - A * x0) if x0 ≠ 0.
  bNorm == 0 && return x, SimpleStats(true, false, [zero(T)], [zero(T)], "x = 0 is a zero-residual solution")
  rNorm = bNorm  # + λ * ‖x0‖ if x0 ≠ 0 and λ > 0.
  λ > 0 && (s = copy(r))
  Aᵀr = Aᵀ * r # - λ * x0 if x0 ≠ 0.
  p .= Aᵀr
  γ = @kdot(n, Aᵀr, Aᵀr)  # Faster than γ = dot(Aᵀr, Aᵀr)
  λ > 0 && (γ += λ * rNorm * rNorm)
  iter = 0
  itmax == 0 && (itmax = m + n)

  ArNorm = sqrt(γ)
  rNorms = history ? [rNorm] : T[]
  ArNorms = history ? [ArNorm] : T[]
  ɛ_c = atol + rtol * rNorm  # Stopping tolerance for consistent systems.
  ɛ_i = atol + rtol * ArNorm  # Stopping tolerance for inconsistent systems.
  (verbose > 0) && @info @sprintf("%5s  %8s  %8s\n", "Aprod", "‖Aᵀr‖", "‖r‖")
  display(iter, verbose) && @info @sprintf("%5d  %8.2e  %8.2e\n", 1, ArNorm, rNorm)

  status = "unknown"
  solved = rNorm ≤ ɛ_c
  inconsistent = (rNorm > 100 * ɛ_c) && (ArNorm ≤ ɛ_i)
  tired = iter ≥ itmax

  while ! (solved || inconsistent || tired)
    q = A * p
    λ > 0 && @kaxpy!(m, λ, s, q)  # q = q + λ * s
    Mq = M * q
    α = γ / @kdot(m, q, Mq)    # Compute qᵗ * M * q
    @kaxpy!(n,  α, p, x)       # Faster than  x =  x + α *  p
    @kaxpy!(m, -α, Mq, r)      # Faster than  r =  r - α * Mq
    rNorm = @knrm2(m, r)       # norm(r)
    Aᵀr = Aᵀ * r
    γ_next = @kdot(n, Aᵀr, Aᵀr)  # Faster than γ_next = dot(Aᵀr, Aᵀr)
    λ > 0 && (γ_next += λ * rNorm * rNorm)
    β = γ_next / γ

    @kaxpby!(n, one(T), Aᵀr, β, p)  # Faster than  p = Aᵀr + β * p
    if λ > 0
      @kaxpby!(m, one(T), r, β, s) # s = r + β * s
    end

    γ = γ_next
    ArNorm = sqrt(γ)
    history && push!(rNorms, rNorm)
    history && push!(ArNorms, ArNorm)
    iter = iter + 1
    display(iter, verbose) && @info @sprintf("%5d  %8.2e  %8.2e\n", 1 + 2 * iter, ArNorm, rNorm)
    solved = rNorm ≤ ɛ_c
    inconsistent = (rNorm > 100 * ɛ_c) && (ArNorm ≤ ɛ_i)
    tired = iter ≥ itmax
  end
  (verbose > 0) && @printf("\n")

  status = tired ? "maximum number of iterations exceeded" : (inconsistent ? "system probably inconsistent but least squares/norm solution found" : "solution good enough given atol and rtol")
  stats = SimpleStats(solved, inconsistent, rNorms, ArNorms, status)
  return (x, stats)
end
