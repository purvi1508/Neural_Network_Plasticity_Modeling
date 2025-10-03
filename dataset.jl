using Revise
using AdFem
using SparseArrays
using LinearAlgebra
using DelimitedFiles

αm = 0.0
αf = 0.0
β2 = 0.5*(1 - αm + αf)^2
γ = 0.5 - αm + αf

m = 40
n = 20
h = 0.01
NT = 100
Δt = 1/NT 
bdedge = []
for j = 1:n 
    push!(bdedge, [(j-1)*(m+1)+m+1 j*(m+1)+m+1])
end
bdedge = vcat(bdedge...)

bdnode = Int64[]
for j = 1:n+1
    push!(bdnode, (j-1)*(m+1)+1)
end

M = compute_fem_mass_matrix1(m, n, h)
S = spzeros((m+1)*(n+1), (m+1)*(n+1))
M = [M S; S M]
H = diagm(0=>[1,1,0.5])
K = 0.1
σY = 0.03

state = zeros(2*(m+1)*(n+1), NT+1)
velo = zeros(2*(m+1)*(n+1), NT+1)
acce = zeros(2*(m+1)*(n+1), NT+1)
stress = zeros(NT+1, 4*m*n, 3)
internal_variable = zeros(NT+1, 4*m*n)

t = 0.0
for i = 1:NT 
    @info i
    ##### STEP 1: Computes the external force ####
    T = eval_f_on_boundary_edge((x,y)->0.02*sin(2π*i*Δt), bdedge, m, n, h)
    T = [zeros(length(T)) -T]
    T = compute_fem_traction_term(T, bdedge, m, n, h)
    
    f1 = eval_f_on_gauss_pts((x,y)->0., m, n, h)
    f2 = eval_f_on_gauss_pts((x,y)->0., m, n, h)
    F = compute_fem_source_term(f1, f2, m, n, h)
    fext = F + T

    ##### STEP 2: Extract Variables ####
    u = state[:,i]
    ∂∂u = acce[:,i]
    ∂u = velo[:,i]

    ε0 = eval_strain_on_gauss_pts(u, m, n, h)
    σ0 = stress[i,:,:]
    α0 = internal_variable[i,:]
    
    ##### STEP 3: Newton Iteration ####
    global t += (1 - αf)*Δt
    ∂∂up = ∂∂u[:]
    iter = 0
    while true
        iter += 1
        up = (1 - αf)*(u + Δt*∂u + 0.5 * Δt^2 * ((1 - β2)*∂∂u + β2*∂∂up)) + αf*u
        global fint, stiff, α, σ = compute_planestressplasticity_stress_and_stiffness_matrix(
            up, ε0, σ0, α0, K, σY, H, m, n, h
        )
        res = M * (∂∂up *(1 - αm) + αm*∂∂u) + fint - fext
        A = M*(1 - αm) + (1 - αf) * 0.5 * β2 * Δt^2 * stiff
        A, _ = fem_impose_Dirichlet_boundary_condition(A, bdnode, m, n, h)
        res[[bdnode; bdnode .+ (m+1)*(n+1)]] .= 0.0

        Δ∂∂u = A\res
        ∂∂up -= Δ∂∂u
        err = norm(res)
        if err < 1e-8
            break 
        end
    end
    global t += αf*Δt
    
    ##### STEP 4: Update State Variables ####
    u += Δt * ∂u + Δt^2/2 * ((1 - β2) * ∂∂u + β2 * ∂∂up)
    ∂u += Δt * ((1 - γ) * ∂∂u + γ * ∂∂up)
    stress[i+1,:,:] = σ
    internal_variable[i+1,:] = α
    
    state[:,i+1] = u
    acce[:,i+1] = ∂∂up
    velo[:,i+1] = ∂u
end

# ===== Save stress and strain for alpha > 0 =====
# open("stress_readable.csv", "w") do io
#     writedlm(io, [["timestep", "point", "σ_xx", "σ_yy", "σ_xy", "λ"]])
#     for i = 1:NT+1
#         for j = 1:4*m*n
#             row = [i, j, stress[i, j, 1], stress[i, j, 2], stress[i, j, 3], internal_variable[i, j]]
#             writedlm(io, [row])
#         end
#     end
# end

# open("strain_readable.csv", "w") do io
#     writedlm(io, [["timestep", "point", "ε_xx", "ε_yy", "ε_xy", "λ"]])
#     for i = 1:NT+1
#         ε_i = eval_strain_on_gauss_pts(state[:, i], m, n, h)
#         for j = 1:4*m*n
#             row = [i, j, ε_i[j, 1], ε_i[j, 2], ε_i[j, 3], internal_variable[i, j]]
#             writedlm(io, [row])
#         end
#     end
# end

open("stress_strain_readable.csv", "w") do io
    # header
    writedlm(io, [["timestep","point","σ_xx","σ_yy","σ_xy","ε_xx","ε_yy","ε_xy","λ"]], ',')

    for i = 1:NT+1
        ε_i = eval_strain_on_gauss_pts(state[:, i], m, n, h)

        for j = 1:4*m*n
            λ_curr = internal_variable[i, j]
            λ_prev = (i == 1) ? 0.0 : internal_variable[i-1, j]
            λ = λ_curr - λ_prev   # incremental plastic multiplier

            row = [
                i,                                # timestep
                j,                                # point index
                stress[i, j, 1],                  # σ_xx
                stress[i, j, 2],                  # σ_yy
                stress[i, j, 3],                  # σ_xy
                ε_i[j, 1],                        # ε_xx
                ε_i[j, 2],                        # ε_yy
                ε_i[j, 3],                        # ε_xy
                λ                                # save incremental λ
            ]
            writedlm(io, [row], ',')
        end
    end
end
