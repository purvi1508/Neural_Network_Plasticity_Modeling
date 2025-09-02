using Flux
using Enzyme
using LinearAlgebra
using Random

# 1. Material parameters
E = 210e3 # MPa
ν = 0.3
G = E/(2*(1+ν)) # Shear Modulus
K = E/(3*(1-2*ν)) # Bulk Modulus
Y0 = 350.0 # MPa Initial stress of the material Y0 comes from a tensile test or compression test.
t_star = 1.0 # s
n_exp = 2
NN0 = 1.0 # scaling for NN outputs

println("Material parameters: G=$G, K=$K, Y0=$Y0")

# 2. Material state
mutable struct MaterialState
    εp::Matrix{Float64} # 3x3 plastic strain
    β::Matrix{Float64} # 3x3 backstress
    κ::Float64 # isotropic hardening
end
#Stores internal variables that evolve over time: plastic strain, backstress, and isotropic hardening(represents the uniform expansion of the yield surface in stress space).

function MaterialStateZero()
    #At the very beginning of a simulation, before any load is applied, the material is assumed undeformed and unstressed.
    #Plastic strain εp = 0 → material has not plastically deformed yet.
    #Backstress β = 0 → no internal stress has accumulated.
    #Isotropic hardening κ = 0 → no hardening has occurred.
    MaterialState(zeros(3,3), zeros(3,3), 0.0)
end

# 3. Helper functions
double_dot(A::Matrix{Float64}, B::Matrix{Float64}) = sum(A .* B)
#.* is element-wise multiplication in Julia.
#Adds all the numbers in the resulting matrix.
#double contraction in tensor #https://github.com/KeitaNakamura/Tensorial.jl ⊡₂

function deviatoric(T::Matrix{Float64})
    return T .- (tr(T)/3.0) * Matrix{Float64}(I, 3, 3)
end
#remove the mean/volumetric part of a tensor → leave only shear/deviatoric part.
#Essential for plasticity models because plastic deformation depends on deviatoric stress, not hydrostatic pressure.

function yield_vonMises(σ::Matrix{Float64}, β::Matrix{Float64}, κ::Float64, Y0::Float64)
    #Von Mises yield function with kinematic (β) and isotropic (κ) hardening:
    #Φ = sqrt( (3/2 )* dev((σ - β) : dev(σ - β))) - [Y0 + κ]
    #Returns > 0 if yielding occurs, ≤ 0 if elastic.
    s=σ - β
    s_dev = deviatoric(s)
    Φ = sqrt((3/2) * double_dot(s, s_dev)) - (Y0 + κ)
    return Φ
end

function flow_direction(σ::Matrix{Float64}, β::Matrix{Float64})
    #ν = (3/2)*((dev(σ) - dev(β)) / f_vM (σ-β))
    #ν=(∂σ/∂Φ)​ = plastic flow direction (normal to yield surface, associative plasticity).
    s=σ - β
    s_dev=deviatoric(s)
    σ_dev=deviatoric(σ)
    β_dev=deviatoric(β)
    f_vM = sqrt((3/2) * double_dot(s, s_dev))
    ν = (3/2) * ((σ_dev-β_dev)/f_vM)
    return ν
end

function compute_invariants(state::MaterialState, ν::Matrix{Float64}, NN0::Float64)
    
    # Extract current material state variables
    β = state.β      # backstress tensor (3x3)
    εp = state.εp    # plastic strain tensor (3x3)
    κ = state.κ      # isotropic hardening (scalar)

    # ----------------------------
    # 1. Invariants as per Eq. (28)
    # ----------------------------
    # I1 = κ (isotropic hardening), scaled by NN0
    I1 = κ / NN0
    # I2 = β : β (magnitude of backstress), scaled by NN0^2
    I2 = double_dot(β, β) / NN0^2
    # I3 = ν : β (interaction between flow direction and backstress), scaled by NN0
    I3 = double_dot(ν, β) / NN0
    # I4 = β : εp (interaction between backstress and plastic strain), scaled by NN0^2
    I4 = double_dot(β, εp) / NN0^2
    # I5 = ν : εp (interaction between flow direction and plastic strain), scaled by NN0
    I5 = double_dot(ν, εp) / NN0
    # I6 = εp : εp (magnitude of plastic strain), scaled by NN0^2
    I6 = double_dot(εp, εp) / NN0^2
    return [I1, I2, I3, I4, I5, I6]
end


function evolution_laws(state::MaterialState, ν::Matrix{Float64}, λdot::Float64, NN_output::Vector{Float64}, NN0::Float64)
    #Proposed evolution laws using NN outputs:
    #b˙=−λ˙[(1−(ν:β) NNk,ν)ν−NNk,ββ]
    #κ˙=−λ˙[1−κ NN iso]
    NN_iso, NN_kβ, NN_kν = NN_output[1:3] ./ NN0
    tr_dot = double_dot(ν, state.β)
    β_dot = -λdot * ((1 - tr_dot * NN_kν) * ν - NN_kβ * state.β)
    κ_dot = λdot * (1 + NN_iso)  # isotropic hardening
    return β_dot, κ_dot
end

# 4. Elastic stress 

function elastic_stress(ε::Matrix{Float64}, state::MaterialState, G::Float64, K::Float64)
    # Elastic stiffness tensor (E):
    # relates stress and strain via:
    #     σ = C : ε
    # where ":" denotes the double-dot product.
    #
    # For a 3D isotropic material:
    #   - σ = stress tensor (3x3)
    #   - ε = strain tensor (3x3)
    #   - C encodes the material's elastic properties.
    #
    # The stress-strain relation can be written as:
    #     σ = 2*G*ε_dev + K*tr(ε)*I
    # shear stress (distortional part) + volumetric stress (hydrostatic part),
    # where:
    #   - G is the shear modulus
    #   - K is the bulk modulus
    #   - ε_dev = deviatoric part of strain = ε - (tr(ε)/3)*I
    #   - tr(ε) = trace of strain (volumetric part)
    #   - I = 3x3 identity matrix
    #For an isotropic material, the elasticity matrix in Voigt notation is: https://doc.comsol.com/5.5/doc/com.comsol.help.sme/sme_ug_theory.06.23.html
    # This formula is equivalent to the derivative of the elastic energy density:
    #     Ψ_e = 1/2 * (ε - ε_p) : C : (ε - ε_p)
    # with respect to ε, where ε_p is the plastic strain.

    εe = ε - state.εp
    return 2*G*deviatoric(εe) + K*tr(εe)*Matrix{Float64}(I,3,3)
end

# 5. Time integration (3D)

function integrate_material_3D(nn, ε_hist_3D, dt, G, K, Y0, n, t_star, NN0)
    # Inputs:
    # nn → the neural network that predicts evolution law parameters from invariants.
    # ε_hist_3D → history of strain tensors over time (3×3 matrices).
    # dt → time step increment.
    # G, K → shear and bulk modulus (elastic properties).
    # Y0 → initial yield stress.
    # n → exponent for the viscoplastic flow rate.
    # t_star → characteristic time.
    # NN0 → scaling factor for neural network outputs.
    # Output:
    # σ_hist → list of stress tensors computed at each time step.
    state = MaterialStateZero()
    #starting material state (plastic strain, backstress, isotropic hardening all zero).
    nt = length(ε_hist_3D)
    σ_hist = [zeros(3,3) for i in 1:nt] #empty array to store the stress at each time step.

    for i in 1:nt
        ε = ε_hist_3D[i]
        #At each step, take the current strain matrix.
        σ = elastic_stress(ε, state, G, K)
        #Compute the elastic stress assuming the current plastic starting
        ν = flow_direction(σ, state.β)
        #Calculates the direction of plastic flow based on deviatoric stress minus backstress
        invariants = compute_invariants(state, ν, NN0)
        #Extract 6 scalar quantities from the current state (plastic strain, backstress, etc.) for the neural network.
        NN_output = Float64.(vec(nn(reshape(Float64.(invariants), :, 1))))
        Φ = yield_vonMises(σ, state.β, state.κ, Y0)
        λdot = max(0, (Φ / Y0)^n / t_star)
        β_dot, κ_dot = evolution_laws(state, ν, λdot, NN_output, NN0)
        state.β .+= β_dot .* dt
        state.κ += κ_dot * dt
        state.εp .+= λdot .* ν .* dt
        σ_hist[i] .= σ
        #Store the current stress tensor in the history array.
    end

    return σ_hist
end


# 6. Neural Network (3D invariants → evolution)

function dense64(in_dim, out_dim, σ)
    W = randn(Float64, out_dim, in_dim)
    b = zeros(Float64, out_dim)
    Dense(W, b, σ)
end

nn = Chain(
    dense64(6,6,tanh),
    dense64(6,6,tanh),
    dense64(6,6,tanh),
    dense64(6,6,tanh),
    dense64(6,5,tanh),
    dense64(5,3,x->x.^2)
)

dup_nn = Enzyme.Duplicated(nn)

# 7. Loss function (3D)

# σ = [σxx σyx σzx; σxy σyy σzy; σxz σyz σzz]
function loss(σ_sim::Vector{Matrix{Float64}}, σ_exp::Vector{Matrix{Float64}}; w_g::Float64=1.0, delta_l::Int=10)
    #we found that a weighting factor 𝑤g = 1 worked well with using data ten points apart to calculate the numerical gradient.

    Nstp = length(σ_sim)           # number of timesteps
    loss_L2 = 0.0                  # first term: normalized L2 loss
    loss_sobolev = 0.0             # second term: derivative/regularization term

    # ---1. Eq. (32) in the paper first part of equation---
    for i in 1:Nstp
        # 1a. Range of experimental stress for normalization
        # If all values in σ_exp[i] are the same, maximum - minimum = 0, add small epsilon to avoid division by zero
        range_exp = maximum(σ_exp[i]) - minimum(σ_exp[i]) + 1e-12
        # 1b. Normalized difference between simulated and experimental stress
        # . and the operator ensures element-wise operation
        normalized_diff=(σ_sim[i].-σ_exp[i])./range_exp
        # 1c. Accumulate squared difference for this timestep
        loss_L2+=sum(normalized_diff.^2)
    end

    # ---1. Eq. (32) in the paper second part of equation---
    # [hat{σ}^s_{ij}]_l → simulated stress at timestep l.
    # [hat{σ}^e_{ij}]_l → experimental stress at timestep l.
    # l+10 → 10 timesteps ahead (in your code: delta_l).
    for l in 1:(Nstp - delta_l)
        # Difference in simulated stress over delta_l steps
        Δσ_sim=σ_sim[l+delta_l].-σ_sim[l]
        # Difference in experimental stress over delta_l steps
        Δσ_exp=σ_exp[l+delta_l].-σ_exp[l]
        # Normalized differences by experimental stress range at timestep l
        range_exp=maximum(σ_exp[l])-minimum(σ_exp[l]) + 1e-12
        Δσ_diff_norm = (Δσ_sim .- Δσ_exp) ./ range_exp
        # Accumulate squared differences scaled by weighting factor w_g
        loss_sobolev += w_g * sum(Δσ_diff_norm.^2)
    end

    # --- 3. Average over timesteps ---
    loss_L2/=Nstp
    loss_sobolev/=max(1, Nstp-delta_l)  # avoid division by zero

    # --- 4. Total loss ---
    total_loss=loss_L2+loss_sobolev

    return total_loss
end


# 8. Training loop (3D)

# nt = 100
# ε_hist_3D = [rand(3,3)*0.01 for i in 1:nt]
# σ_exp_3D = [rand(3,3)*100 for i in 1:nt]

epochs = 10
dt = 0.01 #time increment for material simulation.
#Each step in your strain history is assumed to happen at intervals of dt.

optimizer = Flux.setup(RAdam(1e-3), nn)
#the RAdam optimizer with learning rate 0.001(Since in the paper specific value is not given am using a common value)

for step in 1:epochs
    println("\n=== Training step $step ===")
    #run material simulation using the neural network m
    #strain history ε_hist_3D
    #Neural network outputs are used for evolution laws.
    grads = Flux.gradient((m, ε, σ) -> begin
        σ_sim = integrate_material_3D(m, ε, dt, G, K, Y0, n_exp, t_star, NN0)
        #Returns simulated stress σ_sim for all timesteps.
        loss(σ_sim, σ)
        #compute the difference between simulation and experimental stress.
    end, dup_nn, Const(ε_hist_3D), Const(σ_exp_3D))
    #Gradients tell us how the loss changes if we slightly change each neural network weight.
    #automatically compute gradients of the loss with respect to neural network parameters.

    Flux.update!(optimizer, nn, grads[1])
end
