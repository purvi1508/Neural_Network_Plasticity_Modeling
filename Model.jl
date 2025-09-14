using Flux
using Enzyme
using LinearAlgebra
using Random

# 1. Material parameters
E = 210e3 # MPa Young’s modulus
ν = 0.3 # Poisson’s ratio
G =  Float32(E/(2*(1+ν))) # Shear Modulus
K = Float32(E/(3*(1-2*ν))) # Bulk Modulus
Y0 = Float32(350.0) # MPa Initial stress of the material Y0 comes from a tensile test or compression test.
t_star = Float32(1.0) # _star and n_exp Parameters for viscoplastic flow (flow rate exponent and characteristic time).
n_exp = 1
NN0 = Float32(1.0) # Scaling factor for neural network outputs.

println("Material parameters: G=$G, K=$K, Y0=$Y0")

# 2. Material state
mutable struct MaterialState{T<:AbstractFloat}
    εp::Matrix{T}  # plastic strain
    β::Matrix{T}   # backstress
    κ::T           # isotropic hardening
end

#Stores internal variables that evolve over time: plastic strain, backstress, and isotropic hardening(represents the uniform expansion of the yield surface in stress space).

function MaterialStateZero(::Type{T}=Float32) where T<:AbstractFloat
    #At the very beginning of a simulation, before any load is applied, the material is assumed undeformed and unstressed.
    #Plastic strain εp = 0 → material has not plastically deformed yet.
    #Backstress β = 0 → no internal stress has accumulated.
    #Isotropic hardening κ = 0 → no hardening has occurred.
    MaterialState{T}(zeros(T,3,3), zeros(T,3,3), zero(T))
end

# 3. Helper functions
function double_dot(A::AbstractMatrix{<:AbstractFloat}, B::AbstractMatrix{<:AbstractFloat})
    return Float32(sum(Float32.(A) .* Float32.(B)))
end

#.* is element-wise multiplication in Julia.
#Adds all the numbers in the resulting matrix.
#double contraction in tensor #https://github.com/KeitaNakamura/Tensorial.jl ⊡₂

function deviatoric(T_::AbstractMatrix{T}) where T<:AbstractFloat
    return T_ .- (tr(T_)/3.0) * Matrix{T}(I(3))
end
#remove the mean/volumetric part of a tensor → leave only shear/deviatoric part.
#Essential for plasticity models because plastic deformation depends on deviatoric stress, not hydrostatic pressure.

function yield_vonMises(σ::AbstractMatrix{T}, β::AbstractMatrix{T}, κ::T, Y0::T) where T<:AbstractFloat
    # Von Mises yield function with kinematic (β) and isotropic (κ) hardening
    # Returns >0 if yielding occurs, ≤0 if elastic
    σf = Float32.(σ)
    βf = Float32.(β)

    s = σf .- βf
    s_dev = deviatoric(s)
    σ_eq = sqrt(Float32(3/2) * double_dot(s_dev, s_dev))
    return σ_eq - (Y0 + κ)
end

function flow_direction(σ::AbstractMatrix{T}, β::AbstractMatrix{T}) where T<:AbstractFloat
    #ν = (3/2)*((dev(σ) - dev(β)) / f_vM (σ-β))
    #ν=(∂σ/∂Φ)​ = plastic flow direction (normal to yield surface, associative plasticity).
    σf = Float32.(σ)
    βf = Float32.(β)
    s = σf .- βf
    s_dev = deviatoric(s)
    σ_dev = deviatoric(σf)
    β_dev = deviatoric(βf)
    f_vM = sqrt((3/2) * double_dot(s, s_dev))
    if f_vM < 1e-14
        return zeros(3,3)  # avoid division by zero
    else
        ν = (3/2) * ((σ_dev-β_dev)/f_vM)
        return ν
    end
end

function compute_invariants(state::MaterialState{T}, ν::AbstractMatrix{T}, NN0::T) where T<:AbstractFloat
    # Ensure all state variables are of type T
    β = state.β      # backstress tensor
    εp = state.εp    # plastic strain tensor
    κ = state.κ      # isotropic hardening (scalar)

    # ----------------------------
    # Compute invariants
    # ----------------------------
    I1 = κ / NN0
    I2 = double_dot(β, β) / (NN0^2)
    I3 = double_dot(ν, β) / NN0
    I4 = double_dot(β, εp) / (NN0^2)
    I5 = double_dot(ν, εp) / NN0
    I6 = double_dot(εp, εp) / (NN0^2)

    return T[I1, I2, I3, I4, I5, I6]  # return as vector of type T
end



function evolution_laws(state::MaterialState, ν::Matrix{Float32}, λdot::Float32, NN_output::Vector{Float32}, NN0::Float32)
    #Proposed evolution laws using NN outputs:
    #b˙=−λ˙[(1−(ν:β) NNk,ν)ν−NNk,ββ]
    #κ˙=−λ˙[1−κ NN iso]
    # Neural net outputs
    NN_iso, NN_kβ, NN_kν = NN_output[1:3]

    # Flow interaction
    tr_dot = double_dot(ν, state.β) / NN0   # <-- scale β consistently

    # Backstress evolution (scaled)
    β_dot = -λdot * (((1 - tr_dot * NN_kν) * ν) - (NN_kβ * state.β / NN0))

    # Isotropic hardening (scaled)
    κ_dot = λdot * (1 + NN_iso) * NN0
    return β_dot, κ_dot
end

# 4. Elastic stress 

function elastic_stress(ε::AbstractMatrix{T}, state::MaterialState{T}, G::T, K::T) where T<:AbstractFloat
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

    εe = ε .- state.εp
    return 2*G*deviatoric(εe) + K*tr(εe)*Matrix{T}(I(3))
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
        println("\n===============================")
        println("🔹 Time step: $i / $nt")
        println("===============================")

        # --- Strain history ---
        ε = ε_hist_3D[i]
        println("ε (strain tensor):\n$ε")

        # --- Elastic predictor ---
        σ = Float32.(elastic_stress(ε, state, G, K))
        println("σ (elastic stress tensor):\n$σ")

        # --- Flow direction ---
        ν = Float32.(flow_direction(σ, state.β))
        println("ν (flow direction):\n$ν")

        # --- Invariants for NN input ---
        invariants = compute_invariants(state, ν, NN0)
        println("Invariants (NN input): ", invariants)

        # --- Neural network output ---
        NN_input = reshape(Float32.(invariants), :, 1)
        NN_output = Float32.(vec(nn(Float32.(NN_input))))
        println("NN input (reshaped): ", NN_input)
        println("NN output: ", NN_output)

        # --- Yield function ---
        Φ = yield_vonMises(σ, state.β, state.κ, Y0)
        println("Yield function Φ: ", Φ)

        # --- Plastic multiplier ---
        λdot = max(0, (Φ / Y0)^n / t_star)
        println("λ̇ (plastic multiplier rate): ", λdot)

        # --- Evolution laws ---
        β_dot, κ_dot = evolution_laws(state, ν, λdot, NN_output, NN0)
        println("β̇ (backstress rate):\n$β_dot")
        println("κ̇ (hardening variable rate): ", κ_dot)

        # --- State updates ---
        state.β .+= β_dot .* dt
        state.κ  += κ_dot * dt
        state.εp .+= λdot .* ν .* dt
        println("Updated backstress β:\n", state.β)
        println("Updated κ: ", state.κ)
        println("Updated plastic strain εp:\n", state.εp)

        # --- Store history ---
        σ_hist[i] .= σ
        println("Stored σ_hist[$i]:\n", σ_hist[i])
    end

    return σ_hist

end


# 6. Neural Network (3D invariants → evolution)
nn = Chain(
    Dense(6,6,tanh),
    Dense(6,6,tanh),
    Dense(6,6,tanh),
    Dense(6,6,tanh),
    Dense(6,5,tanh),
    Dense(5,3,x->x.^2) #One Dense(5 → 3, x -> x.^2) → output layer with 3 neurons, using squared activation
)
# W = weight matrix, shape (out_dim, in_dim), initialized with standard normal values.
# b = bias vector of length out_dim, initialized to zeros.
# Returns a Dense layer (from Flux.jl) with these weights, biases, and the specified activation.

dup_nn = Enzyme.Duplicated(nn)

# 7. Loss function (3D)

# σ = [σxx σyx σzx; σxy σyy σzy; σxz σyz σzz]
function loss_exact_paper(σ_sim::Vector{<:AbstractMatrix{<:AbstractFloat}}, 
                          σ_exp::Vector{<:AbstractMatrix{<:AbstractFloat}};
                          w_g::Float32=Float32(1.0), delta_l::Int=10)
    Nstp = length(σ_sim)
    ncomp = size(σ_sim[1], 1)

    # Compute global stress range
    smin = minimum([σ_exp[l][a,b] for l in 1:Nstp, a in 1:ncomp, b in 1:ncomp])
    smax = maximum([σ_exp[l][a,b] for l in 1:Nstp, a in 1:ncomp, b in 1:ncomp])
    range_exp = smax - smin + 1e-12

    # Normalized L2 loss
    loss_L2 = 0.0
    for l in 1:Nstp
        for i in 1:ncomp, j in 1:ncomp
            d = (Float32(σ_sim[l][i,j]) - Float32(σ_exp[l][i,j])) / (sqrt(Nstp) * Float32(range_exp))
            loss_L2 += d^2
        end
    end

    # Sobolev term
    loss_sobolev = 0.0
    for l in 1:(Nstp - delta_l)
        for i in 1:ncomp, j in 1:ncomp
            Δsim = Float32(σ_sim[l+delta_l][i,j]) - Float32(σ_sim[l][i,j])
            Δexp = Float32(σ_exp[l+delta_l][i,j]) - Float32(σ_exp[l][i,j])
            d = (Δsim - Δexp) / (sqrt(Nstp) * Float32(range_exp))
            loss_sobolev += w_g * d^2
        end
    end
    tot_loss=loss_L2 + loss_sobolev
    println("\n=== Loss $tot_loss ===")
    return tot_loss
end


# Lame constants
λ = (E*ν) / ((1+ν)*(1-2ν))
μ = E / (2*(1+ν))   # shear modulus

# build stiffness tensor C (Voigt 6x6 form)
C_voigt = [
    λ+2μ   λ      λ      0     0     0
    λ     λ+2μ    λ      0     0     0
    λ      λ     λ+2μ    0     0     0
    0      0      0      μ     0     0
    0      0      0      0     μ     0
    0      0      0      0     0     μ
]

# helper to convert strain tensor (3x3) to Voigt vector (6x1)
function tensor_to_voigt(ε::AbstractMatrix{T}) where T<:AbstractFloat
    return T[
        ε[1,1];
        ε[2,2];
        ε[3,3];
        2*ε[2,3];
        2*ε[1,3];
        2*ε[1,2];
    ]
end

# and back: Voigt vector → 3x3 stress tensor
function voigt_to_tensor(σv::AbstractVector{T}) where T<:AbstractFloat
    return T[
        σv[1]  σv[6]/2  σv[5]/2;
        σv[6]/2 σv[2]   σv[4]/2;
        σv[5]/2 σv[4]/2 σv[3]
    ]
end

# Strain history (uniaxial loading in z-direction) simulates a uniaxial tensile test.
nt = 30
ε_hist_3D = [Float32.(Matrix(Diagonal([0.0, 0.0, ε]))) for ε in range(0, 0.02, length=nt)]
println("\n=== Strain $ε_hist_3D ===")
# Compute consistent stresses
C_voigt = Float32.(C_voigt)
σ_exp_3D = [Float32.(voigt_to_tensor(C_voigt * tensor_to_voigt(ε))) for ε in ε_hist_3D]
println("\n=== Stress $σ_exp_3D ===")

epochs = 10
dt = 0.01 #time increment for material simulation.
#Each step in your strain history is assumed to happen at intervals of dt.

optimizer = Flux.setup(RAdam(1e-3), nn)
#the RAdam optimizer with learning rate 0.001(Since in the paper specific value is not given am using a common value)

for step in 1:epochs
    println("\n=== Training step $step ===")
    
    grads = Flux.gradient((m, ε, σ) -> 
        loss_exact_paper(
            integrate_material_3D(m, ε, dt, G, K, Y0, n_exp, t_star, NN0),
            σ
        ), 
        dup_nn, Const(ε_hist_3D), Const(σ_exp_3D)
    )
    
    # Update original neural network
    Flux.update!(optimizer, nn, grads[1])
end
