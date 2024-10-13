# An implementation of USYMLQR for the solution of symmetric saddle-point systems.
#
# This method is described in
#
# M. A. Saunders, H. D. Simon, and E. L. Yip
# Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations.
# SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
#
# A. Buttari, D. Orban, D. Ruiz and D. Titley-Peloquin
# A tridiagonalization method for symmetric saddle-point systems.
# SIAM Journal on Scientific Computing, 41(5), pp. 409--432, 2019
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Alexis Montoison, <alexis.montoison@polymtl.ca> -- <amontoison@anl.gov>
# Montréal, November 2021 -- Chicago, October 2024.

export usymlqr, usymlqr!

"""
   (x, y, stats) = usymlqr(A, b::AbstractVector{FC}, c::AbstractVector{FC};
                           M=I, N=I, ldiv::Bool=false, atol::T=√eps(T),
                           rtol::T=√eps(T), itmax::Int=0, timemax::Float64=Inf,
                           verbose::Int=0, history::Bool=false,
                           callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, y, stats) = usymlqr(A, b, c, x0::AbstractVector, y0::AbstractVector; kwargs...)

USYMLQR can be warm-started from initial guesses `x0` and `y0` where `kwargs` are the same keyword arguments as above.

Solve the symmetric saddle-point system

    [ E   A ] [ x ] = [ b ]
    [ Aᴴ    ] [ y ]   [ c ]

where E = M⁻¹ ≻ 0 by way of the Saunders-Simon-Yip tridiagonalization using USYMLQ and USYMQR methods.
The method solves the least-squares problem

    [ E   A ] [ s ] = [ b ]
    [ Aᴴ    ] [ t ]   [ 0 ]

and the least-norm problem

    [ E   A ] [ w ] = [ 0 ]
    [ Aᴴ    ] [ z ]   [ c ]

and simply adds the solutions.

    [ M   O ]
    [ 0   N ]

indicates the weighted norm in which residuals are measured.
It's the Euclidean norm when `M` and `N` are identity operators.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension m × n;
* `b`: a vector of length m;
* `c`: a vector of length n.

#### Optional arguments

* `x0`: a vector of length m that represents an initial guess of the solution x;
* `y0`: a vector of length n that represents an initial guess of the solution y.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `m` used for centered preconditioning of the partitioned system;
* `N`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning of the partitioned system;
* `ldiv`: define whether the preconditioners use `ldiv!` or `mul!`;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `m+n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be kdisplayed if verbose mode is enabled (verbose > 0). Information will be kdisplayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(solver)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length m;
* `y`: a dense vector of length n;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### References

* M. A. Saunders, H. D. Simon, and E. L. Yip, [*Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations*](https://doi.org/10.1137/0725052), SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
* A. Buttari, D. Orban, D. Ruiz and D. Titley-Peloquin, [*A tridiagonalization method for symmetric saddle-point and quasi-definite systems*](https://doi.org/10.1137/18M1194900), SIAM Journal on Scientific Computing, 41(5), pp. 409--432, 2019.
"""
function usymlqr end

"""
    solver = usymlqr!(solver::UsymlqrSolver, A, b, c; kwargs...)
    solver = usymlqr!(solver::UsymlqrSolver, A, b, c, x0, y0; kwargs...)

where `kwargs` are keyword arguments of [`usymlqr`](@ref).

See [`UsymlqrSolver`](@ref) for more details about the `solver`.
"""
function usymlqr! end

def_args_usymlqr = (:(A                    ),
                    :(b::AbstractVector{FC}),
                    :(c::AbstractVector{FC}))

def_optargs_usymlqr = (:(x0::AbstractVector),
                       :(y0::AbstractVector))

def_kwargs_usymlqr = (:(; M = I                     ),
                      :(; N = I                     ),
                      :(; ldiv::Bool = false        ),
                      :(; atol::T = √eps(T)         ),
                      :(; rtol::T = √eps(T)         ),
                      :(; itmax::Int = 0            ),
                      :(; timemax::Float64 = Inf    ),
                      :(; verbose::Int = 0          ),
                      :(; history::Bool = false     ),
                      :(; callback = solver -> false),
                      :(; iostream::IO = kstdout    ))

def_kwargs_usymlqr = mapreduce(extract_parameters, vcat, def_kwargs_usymlqr)

args_usymlqr = (:A, :b, :c)
optargs_usymlqr = (:x0, :y0)
kwargs_usymlqr = (:M, :N, :ldiv, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function usymlqr!(solver :: UsymlqrSolver{T,FC,S}, $(def_args_usymlqr...); $(def_kwargs_usymlqr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    length(b) == m || error("Inconsistent problem size")
    length(c) == n || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "USYMLQR: system of %d equations in %d variables\n", m+n, m+n)

    # Check M = Iₘ and N = Iₙ
    MisI = (M === I)
    NisI = (N === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")
    ktypeof(c) <: S || error("ktypeof(c) is not a subtype of $S")

    # Compute the adjoint of A
    Aᴴ = A'

    # Set up workspace.
    allocate_if(!MisI, solver, :vₖ, S, m)
    allocate_if(!NisI, solver, :uₖ, S, n)
    M⁻¹vₖ₋₁, M⁻¹vₖ, N⁻¹uₖ₋₁, N⁻¹uₖ = solver.M⁻¹vₖ₋₁, solver.M⁻¹vₖ, solver.N⁻¹uₖ₋₁, solver.N⁻¹uₖ
    xₖ, yₖ, p, q = solver.x, solver.y, solver.p, solver.q
    vₖ = MisI ? M⁻¹vₖ : solver.vₖ
    uₖ = NisI ? N⁻¹uₖ : solver.uₖ
    warm_start = solver.warm_start
    b₀ = warm_start ? q : b
    c₀ = warm_start ? p : c

    stats = solver.stats
    rNorms = stats.residuals
    reset!(stats)

    iter = 0
    itmax == 0 && (itmax = n+m)

    # Initial solutions x₀ and y₀.
    warm_start && (Δx .= xₖ)
    warm_start && (Δy .= yₖ)
    xₖ .= zero(T)
    yₖ .= zero(T)

    # Initialize preconditioned orthogonal tridiagonalization process.
    M⁻¹vₖ₋₁ .= zero(T)  # v₀ = 0
    N⁻¹uₖ₋₁ .= zero(T)  # u₀ = 0

    # [ I   A ] [ xₖ ] = [ b -   Δx - AΔy ] = [ b₀ ]
    # [ Aᴴ    ] [ yₖ ]   [ c - AᴴΔx       ]   [ c₀ ]
    if warm_start
      mul!(b₀, A, Δy)
      @kaxpy!(m, one(T), Δx, b₀)
      @kaxpby!(m, one(T), b, -one(T), b₀)
      mul!(c₀, Aᴴ, Δx)
      @kaxpby!(n, one(T), c, -one(T), c₀)
    end

    # β₁Ev₁ = b ↔ β₁v₁ = Mb
    M⁻¹vₖ .= b₀
    MisI || mul!(vₖ, M, M⁻¹vₖ)
    βₖ = sqrt(@kdot(m, vₖ, M⁻¹vₖ))  # β₁ = ‖v₁‖_E
    if βₖ ≠ 0
      @kscal!(m, 1 / βₖ, M⁻¹vₖ)
      MisI || @kscal!(m, 1 / βₖ, vₖ)
    else
      error("b must be nonzero")
    end

    # γ₁Fu₁ = c ↔ γ₁u₁ = Nc
    N⁻¹uₖ .= c₀
    NisI || mul!(uₖ, N, N⁻¹uₖ)
    γₖ = sqrt(@kdot(n, uₖ, N⁻¹uₖ))  # γ₁ = ‖u₁‖_F
    if γₖ ≠ 0
      @kscal!(n, 1 / γₖ, N⁻¹uₖ)
      NisI || @kscal!(n, 1 / γₖ, uₖ)
    else
      error("c must be nonzero")
    end

    (verbose > 0) && @printf(iostream, "%4s %7s %7s %7s\n", "k", "αₖ", "βₖ", "γₖ")
    kdisplay(iter, verbose) && @printf(iostream, "%4d %7.1e %7.1e %7.1e\n", iter, αₖ, βₖ, γₖ)

    # Stopping criterion.
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    status = "unknown"
    ill_cond = false
    user_requested_exit = false
    overtimed = false

    while !(solved || tired || ill_cond || user_requested_exit || overtimed)
      # Update iteration index.
      iter = iter + 1

      # Continue the orthogonal tridiagonalization process.
      # AUₖ  = EVₖTₖ    + βₖ₊₁Evₖ₊₁(eₖ)ᵀ = EVₖ₊₁Tₖ₊₁.ₖ
      # AᴴVₖ = FUₖ(Tₖ)ᴴ + γₖ₊₁Fuₖ₊₁(eₖ)ᵀ = FUₖ₊₁(Tₖ.ₖ₊₁)ᴴ

      mul!(q, A , uₖ)  # Forms Evₖ₊₁ : q ← Auₖ
      mul!(p, Aᴴ, vₖ)  # Forms Fuₖ₊₁ : p ← Aᴴvₖ

      if iter ≥ 2
        @kaxpy!(m, -γₖ, M⁻¹vₖ₋₁, q)  # q ← q - γₖ * M⁻¹vₖ₋₁
        @kaxpy!(n, -βₖ, N⁻¹uₖ₋₁, p)  # p ← p - βₖ * N⁻¹uₖ₋₁
      end

      αₖ = @kdot(m, vₖ, q)  # αₖ = qᴴvₖ

      @kaxpy!(m, -αₖ, M⁻¹vₖ, q)  # q ← q - αₖ * M⁻¹vₖ
      @kaxpy!(n, -αₖ, N⁻¹uₖ, p)  # p ← p - αₖ * N⁻¹uₖ

      # Compute vₖ₊₁ and uₖ₊₁
      MisI || mul!(vₖ₊₁, M, q)  # βₖ₊₁vₖ₊₁ = MAuₖ  - γₖvₖ₋₁ - αₖvₖ
      NisI || mul!(uₖ₊₁, N, p)  # γₖ₊₁uₖ₊₁ = NAᴴvₖ - βₖuₖ₋₁ - αₖuₖ

      βₖ₊₁ = sqrt(@kdot(m, vₖ₊₁, q))  # βₖ₊₁ = ‖vₖ₊₁‖_E
      γₖ₊₁ = sqrt(@kdot(n, uₖ₊₁, p))  # γₖ₊₁ = ‖uₖ₊₁‖_F

      if βₖ₊₁ ≠ 0
        @kscal!(m, one(T) / βₖ₊₁, q)
        MisI || @kscal!(m, one(T) / βₖ₊₁, vₖ₊₁)
      end

      if γₖ₊₁ ≠ 0
        @kscal!(n, one(T) / γₖ₊₁, p)
        NisI || @kscal!(n, one(T) / γₖ₊₁, uₖ₊₁)
      end

      # Continue the QR factorization of Tₖ₊₁.ₖ = Qₖ₊₁ [ Rₖ ].
      #                                                [ Oᴴ ]

      ƛ = -cs * γₖ
      ϵ =  sn * γₖ

      if !solved_LS
        ArNorm_qr_computed = rNorm_qr * sqrt(δbar^2 + ƛ^2)
        ArNorm_qr = norm(A' * (b - A * x))  # FIXME
        @debug "" ArNorm_qr_computed ArNorm_qr abs(ArNorm_qr_computed - ArNorm_qr) / ArNorm_qr
        ArNorm_qr = ArNorm_qr_computed
        push!(ArNorms_qr, ArNorm_qr)

        test_LS = ArNorm_qr / (Anorm * max(one(T), rNorm_qr))
        solved_lim_LS = test_LS ≤ ls_optimality_tol
        solved_mach_LS = one(T) + test_LS ≤ one(T)
        solved_LS = solved_mach_LS | solved_lim_LS
      end
      kdisplay(iter, verbose) && @printf(iostream, "%7.1e ", ArNorm_qr)

      # continue QR factorization
      delta = sqrt(δbar^2 + βₖ^2)
      csold = cs
      snold = sn
      cs = δbar/ delta
      sn = βₖ / delta

      # update w (used to update x and z)
      @. wold = w
      @. w = cs * wbar

      if !solved_LS
        # the optimality conditions of the LS problem were not triggerred
        # update x and see if we have a zero residual

        ϕ = cs * ϕbar
        ϕbar = sn * ϕbar
        @kaxpy!(n, ϕ, w, x)
        xNorm = norm(x)  # FIXME

        # update least-squares residual
        rNorm_qr = abs(ϕbar)
        push!(rNorms_qr, rNorm_qr)

        # stopping conditions related to the least-squares problem
        test_LS = rNorm_qr / (one(T) + Anorm * xNorm)
        zero_resid_lim_LS = test_LS ≤ ls_zero_resid_tol
        zero_resid_mach_LS = one(T) + test_LS ≤ one(T)
        zero_resid_LS = zero_resid_mach_LS | zero_resid_lim_LS
        solved_LS |= zero_resid_LS

      end

      # continue tridiagonalization
      q = A * vₖ
      @. q -= γₖ * u_prev
      αₖ = @kdot(m, uₖ, q)

      # Update norm estimates
      Anorm2 += αₖ * αₖ + βₖ * βₖ + γₖ * γₖ
      Anorm = √Anorm2

      # Estimate κ₂(A) based on the diagonal of L.
      sigma_min = min(delta, sigma_min)
      sigma_max = max(delta, sigma_max)
      Acond = sigma_max / sigma_min

      # continue QR factorization of T{k+1,k}
      λ = cs * ƛ + sn * αₖ
      δbar= sn * ƛ - cs * αₖ

      if !solved_LN

        etaold = η
        η = cs * etabar # = etak

        # compute residual of least-norm problem at y{k-1}
        # TODO: use recurrence formula for LQ residual
        rNorm_lq_computed = sqrt((delta * η)^2 + (ϵ * etaold)^2)
        rNorm_lq = norm(A' * y - c)  # FIXME
        rNorm_lq = rNorm_lq_computed
        push!(rNorms_lq, rNorm_lq)

        # stopping conditions related to the least-norm problem
        test_LN = rNorm_lq / sqrt(cnorm^2 + Anorm2 * yNorm2)
        solved_lim_LN = test_LN ≤ ln_tol
        solved_mach_LN = one(T) + test_LN ≤ one(T)
        solved_LN = solved_lim_LN || solved_mach_LN

        # TODO: remove this when finished
        push!(tests_LN, test_LN)

        @. wbar = (vₖ - λ * w - ϵ * wold) / δbar

        if !solved_LN

            # prepare to update y and z
            @. p = cs * pbar + sn * uₖ

            # update y and z
            @. y += η * p
            @. z -= η * w
            yNorm2 += η * η
            yNorm = sqrt(yNorm2)

            @. pbar = sn * pbar - cs * uₖ
            etabarold = etabar
            etabar = -(λ * η + ϵ * etaold) / δbar # = etabar{k+1}

            # see if CG iterate has smaller residual
            # TODO: use recurrence formula for CG residual
            @. yC = y + etabar * pbar
            @. zC = z - etabar * wbar
            yCNorm2 = yNorm2 + etabar* etabar
            rNorm_cg_computed = γₖ * abs(snold * etaold - csold * etabarold)
            rNorm_cg = norm(A' * yC - c)

            # if rNorm_cg < rNorm_lq
            #   # stopping conditions related to the least-norm problem
            # test_cg = rNorm_cg / sqrt(γ₁^2 + Anorm2 * yCNorm2)
            #   solved_lim_LN = test_cg ≤ ln_tol
            #   solved_mach_LN = 1.0 + test_cg ≤ 1.0
            #   solved_LN = solved_lim_LN | solved_mach_LN
            #   # transition_to_cg = solved_LN
            #   transition_to_cg = false
            # end

            if transition_to_cg
              # @. yC = y + etabar* pbar
              # @. zC = z - etabar* wbar
            end
        end
      end
      kdisplay(iter, verbose) && @printf(iostream, "%7.1e\n", rNorm_lq)
      kdisplay(iter, verbose) && @printf(iostream, "%4d %8.1e %7.1e %7.1e %7.1e %7.1e %7.1e ", iter, αₖ, βₖ, γₖ, Anorm, Acond, rNorm_qr)

      # Stopping conditions that apply to both problems
      ill_cond_lim = one(T) / Acond ≤ ctol
      ill_cond_mach = one(T) + one(T) / Acond ≤ one(T)
      ill_cond = ill_cond_mach || ill_cond_lim
      tired = iter ≥ itmax
      solved = solved_LS && solved_LN
      user_requested_exit = callback(solver) :: Bool
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Update status
    tired               && (status = "maximum number of iterations exceeded")
    solved              && (status = "solution good enough given atol and rtol")
    ill_cond_mach       && (status = "condition number seems too large for this machine")
    ill_cond_lim        && (status = "condition number exceeds tolerance")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update x and y
    warm_start && @kaxpy!(m, one(FC), Δx, xₖ)
    warm_start && @kaxpy!(n, one(FC), Δy, yₖ)

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = false
    stats.timer = ktimer(start_time)
    stats.status = status
    return solver
  end
end
