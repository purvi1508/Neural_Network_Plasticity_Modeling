using CSV, DataFrames, Flux, Enzyme, LinearAlgebra, Statistics

# -------------------------------
# 1. Load data
df = CSV.read("strain_with_plastic_NN_inputs.csv", DataFrame)

nn_input_cols  = [:I1, :I2, :I3, :I4, :I5, :I6]
nn_output_cols = [:NNiso, :NNk_beta, :NNk_nu]

grouped = groupby(df, :point)

# -------------------------------
# 2. Build training data
nn_data_by_point = Dict{Int, NamedTuple}()
for g in grouped
    pid = g.point[1]
    X      = Float32.(Matrix(g[:, nn_input_cols]))
    Y      = Float32.(Matrix(g[:, nn_output_cols]))
    λ_hist = Float32.(g.λ)
    σ_exp  = Float32.(Matrix(g[:, [:σ_xx, :σ_yy, :σ_xy]]))
    ε      = Float32.(Matrix(g[:, [:ε_xx, :ε_yy, :ε_xy]]))
    nn_data_by_point[pid] = (X=X, Y=Y, λ=λ_hist, σ_exp=σ_exp, ε=ε)
end

# -------------------------------
# 3. Define NN model
last_activation(x) = x.^2f0   # ensures non-negative outputs

model = Chain(
    Dense(6, 6, tanh),
    Dense(6, 6, tanh),
    Dense(6, 6, tanh),
    Dense(6, 5, tanh),
    Dense(5, 3, last_activation)
)

dup_model = Enzyme.Duplicated(model)
opt_state = Flux.setup(Adam(0.001), dup_model)

# -------------------------------
# 4. Material constants
E     = Float32(210e3)  
ν_mat = 0.3f0
Y0    = 100f0

# Plane stress compliance
function elastic_compliance(E::Float32, ν::Float32)
    C = zeros(Float32, 3, 3)
    factor = E / (1f0 - ν^2)
    C[1,1] = 1f0; C[1,2] = ν; C[2,1] = ν; C[2,2] = 1f0
    C[3,3] = (1f0 - ν)/2f0
    return factor * C
end

C_elastic = elastic_compliance(E, ν_mat)

# -------------------------------
# 5. Helper functions
I3f = Matrix{Float32}(I, 3,3)

double_dot(A::AbstractMatrix, B::AbstractMatrix) = Float32(sum(A .* B))
deviatoric(T::AbstractMatrix) = T .- (tr(T)/3f0) .* I3f

flow_direction(σ::AbstractMatrix, β::AbstractMatrix) = begin
    s_dev = deviatoric(σ .- β)
    f_vM = sqrt(3f0/2f0 * double_dot(s_dev, s_dev))
    f_vM > 0f0 ? (3f0/2f0) .* (s_dev / f_vM) : zeros(Float32, 3,3)
end

function evolution_laws(β, ν, λdot, κ, NN_output)
    NN_iso, NN_kβ, NN_kν = NN_output
    tr_dot = double_dot(ν, β)
    β_dot = -λdot * (((1f0 - tr_dot * NN_kν) .* ν) .- (NN_kβ .* β))
    κ_dot = -λdot * (1f0 - κ * NN_iso)
    return β_dot, κ_dot
end

sigma_plane_stress(ε, εp, C_elastic) = C_elastic * (ε .- [εp[1,1]; εp[2,2]; εp[1,2]])

# -------------------------------
# 7. Normalize stress
function normalize_stress(σ, σ_exp)
    n_steps = size(σ_exp,2)
    σ_norm = similar(σ)
    for i in 1:3
        srange = maximum(σ_exp[i,:]) - minimum(σ_exp[i,:])
        σ_norm[i,:] = σ[i,:] ./ (sqrt(n_steps) * srange)
    end
    return σ_norm
end

function compute_sigma_vec(NNiso, NNk_beta, NNk_nu, ε, λ_hist, C_elastic)
    n_samples = length(λ_hist)
    
    # Preallocate arrays
    σ = zeros(Float32, 3, n_samples)
    β = zeros(Float32, 3,3)
    κ = 0f0
    εp = zeros(Float32,3,3)
    σ_prev_tensor = zeros(Float32,3,3)

    for l in 1:n_samples
        λdot = λ_hist[l]

        # flow direction
        ν = flow_direction(σ_prev_tensor, β)

        # evolution
        β_dot, κ_dot = evolution_laws(β, ν, λdot, κ, [NNiso[l], NNk_beta[l], NNk_nu[l]])
        β .+= β_dot
        κ += κ_dot
        εp .+= λdot .* ν

        # plane stress update (vectorized for 3x1)
        ε_vec = ε[:,l]
        β_vec = [β[1,1]; β[2,2]; β[1,2]]
        σ[:,l] = C_elastic * (ε_vec .- β_vec)

        # update previous stress tensor
        σ_prev_tensor = [σ[1,l] σ[3,l] 0f0; σ[3,l] σ[2,l] 0f0; 0f0 0f0 0f0]
    end
    return σ
end
# -------------------------------
# 8. Gradient penalty
function gradient_penalty(σ_sim, σ_exp, step_gap::Int=10)
    n_steps = size(σ_sim,2)
    penalty = 0f0
    for i in 1:3
        for l in 1:(n_steps - step_gap)
            Δ_sim = σ_sim[i, l+step_gap] - σ_sim[i,l]
            Δ_exp = σ_exp[i, l+step_gap] - σ_exp[i,l]
            penalty += (Δ_sim - Δ_exp)^2
        end
    end
    return penalty / (n_steps - step_gap)
end

# -------------------------------
# 9. Stress loss function
function stress_loss(model, X, σ_exp, ε, λ_hist)
    w_g = 1f0
    Y_pred = model(X)  
    NNiso    = Y_pred[1, :]
    NNk_beta = Y_pred[2, :]
    NNk_nu   = Y_pred[3, :]

    σ_sim = compute_sigma_vec(NNiso, NNk_beta, NNk_nu, ε, λ_hist, C_elastic)
    σ_sim_norm = normalize_stress(σ_sim, σ_exp)
    σ_exp_norm = normalize_stress(σ_exp, σ_exp)

    L_L2 = sum((σ_sim_norm .- σ_exp_norm).^2)/size(σ_exp,2)
    L_grad = w_g * gradient_penalty(σ_sim_norm, σ_exp_norm)
    
    return L_L2 + L_grad
end

# -------------------------------
# 10. Training loop
epochs = 10
for (pid, data) in nn_data_by_point
    X      = Float32.(data.X)'
    ε      = Float32.(data.ε)'
    σ_exp  = Float32.(data.σ_exp)'
    λ_hist = Float32.(data.λ)

    println("Training point $pid")

    for epoch in 1:epochs
        grads = Flux.gradient(stress_loss, dup_model, Const(X), Const(σ_exp), Const(ε), Const(λ_hist))
        Flux.update!(opt_state, model, grads[1])
    end
end

println("Training complete.")
