using CSV, DataFrames, LinearAlgebra, Statistics

# -------------------------
# 1. Read CSVs
df_strain = CSV.read("stress_strain_readable.csv", DataFrame)

# -------------------------
# 2. Material properties
E     = 210e3     # MPa
ν_mat = 0.3
Hkin  = 500e3    # MPa
Hiso  = 25e3     # MPa
Y0    = 100      # MPa
NN0   = 1000.0   # scaling for NN inputs
β_inf = 500   # saturation for kinematic hardening
κ_inf = 100   # saturation for isotropic hardening
NNiso  = zeros(nrows)
NNk_beta = zeros(nrows)
NNk_nu   = zeros(nrows)
# -------------------------
# 3. Check 2D or 3D
is3D = (:ε_zz in names(df_strain)) && (:σ_zz in names(df_strain))
ncomp = is3D ? 6 : 3

# -------------------------
# 4. Elastic stiffness
C = zeros(ncomp,ncomp)
if is3D
    C[1,1] = E*(1-ν_mat)/((1+ν_mat)*(1-2*ν_mat))
    C[2,2] = C[1,1]; C[3,3] = C[1,1]
    C[1,2] = E*ν_mat/((1+ν_mat)*(1-2*ν_mat))
    C[1,3] = C[1,2]; C[2,1] = C[1,2]; C[2,3] = C[1,2]; C[3,1] = C[1,2]; C[3,2] = C[1,2]
    C[4,4] = E/(2*(1+ν_mat)); C[5,5] = C[4,4]; C[6,6] = C[4,4]
else
    C[1,1] = E/(1-ν_mat^2); C[2,2] = C[1,1]
    C[1,2] = E*ν_mat/(1-ν_mat^2); C[2,1] = C[1,2]
    C[3,3] = E/(2*(1+ν_mat))
end
S = inv(C)

# -------------------------
# 5. Helper functions

function double_dot(A, B)
    return Float32(sum(A .* B))
end

function deviatoric(T_) 
    I3f = Matrix{Float32}(I, 3, 3)
    return T_ .- (tr(T_)/3f0) .* I3f
end

function yield_vonMises(σ, β, κ, Y0) 
    s=(Float32.(σ .- β))
    s_dev = deviatoric(Float32.(σ .- β))
    σ_eq = sqrt(Float32(3/2) * double_dot(s, s_dev)) 
    return σ_eq - (Y0 + κ)
end

function flow_direction(σ, β) 
    s=(Float32.(σ .- β))
    s_dev = deviatoric(Float32.(σ .- β))
    f_vM = sqrt(Float32(3/2) * double_dot(s, s_dev))
    return f_vM > 0 ? (3/2) * (s_dev / f_vM) : zeros(Float32, size(σ))
end

# -------------------------
# 6. Prepare storage
nrows = size(df_strain,1)
εp = zeros(nrows,ncomp)
β  = zeros(nrows,ncomp)
κ  = zeros(nrows)
I1 = zeros(nrows); I2 = zeros(nrows); I3 = zeros(nrows)
I4 = zeros(nrows); I5 = zeros(nrows); I6 = zeros(nrows)
bdot = zeros(nrows, ncomp)   # backstress rate
kdot = zeros(nrows)          # isotropic hardening rate
ν_storage  = zeros(nrows, ncomp)
# -------------------------
# 7. Loop over points
points = unique(df_strain.point)

for p in points
    idxs = findall(df_strain.point .== p)
    sort_idx = sortperm(df_strain.timestep[idxs])
    idxs = idxs[sort_idx]

    b_prev = zeros(3,3)
    k_prev = 0.0

    for i_global in idxs
        λ_val = df_strain.λ[i_global]
        # Build strain & stress vectors
        if is3D
            ε_vec = [df_strain.ε_xx[i_global], df_strain.ε_yy[i_global], df_strain.ε_zz[i_global],
                     df_strain.ε_yz[i_global], df_strain.ε_xz[i_global], df_strain.ε_xy[i_global]]
            σ_vec = [df_strain.σ_xx[i_global], df_strain.σ_yy[i_global], df_strain.σ_zz[i_global],
                     df_strain.σ_yz[i_global], df_strain.σ_xz[i_global], df_strain.σ_xy[i_global]]
            σ_tensor = [σ_vec[1] σ_vec[6] σ_vec[5];
                        σ_vec[6] σ_vec[2] σ_vec[4];
                        σ_vec[5] σ_vec[4] σ_vec[3]]
        else
            ε_vec = [df_strain.ε_xx[i_global], df_strain.ε_yy[i_global], df_strain.ε_xy[i_global]]
            σ_vec = [df_strain.σ_xx[i_global], df_strain.σ_yy[i_global], df_strain.σ_xy[i_global]]
            σ_tensor = [σ_vec[1] σ_vec[3] 0.0;
                        σ_vec[3] σ_vec[2] 0.0;
                        0.0      0.0      0.0]
        end

        # Plastic strain
        εp[i_global,:] = ε_vec - S*σ_vec

        # Yield function
        κ_val = -abs(Hiso) * k_prev
        Φ = yield_vonMises(σ_tensor, -2/3*Hkin*b_prev, κ_val, Y0)

        # Plastic rates
        if λ_val == 0.0 
            bdot_tensor = zeros(3,3)
            kdot_val = 0.0
            ν_tensor = zeros(3,3)
            NNiso[i_global] = 0
            NNk_beta[i_global] = 0
            NNk_nu[i_global] = 0
        else
            ν_tensor = flow_direction(σ_tensor, -2/3*Hkin*b_prev)
            bdot_tensor = -λ_val * (ν_tensor + (b_prev / β_inf))
            kdot_val = -λ_val * (1.0 - k_prev / κ_inf)
            b_prev .+= bdot_tensor
            k_prev += kdot_val
            NNiso[i_global] =((kdot_val / (λ_val))+1)/NN0
            NNk_beta[i_global] = (sum(bdot_tensor .* (-2/3*Hkin*b_prev)) / (sum((-2/3*Hkin*b_prev).^2)))/NN0
            NNk_nu[i_global] = (sum(bdot_tensor .* ν_tensor) / (sum(ν_tensor .* (-2/3*Hkin*b_prev))))/NN0
        end

        # Store backstress, flow direction, rates
        if is3D
            β[i_global,:]  = [-2/3*Hkin*b_prev[1,1], -2/3*Hkin*b_prev[2,2], -2/3*Hkin*b_prev[3,3],
                              -2/3*Hkin*b_prev[2,3], -2/3*Hkin*b_prev[1,3], -2/3*Hkin*b_prev[1,2]]
            bdot_vec = [bdot_tensor[1,1], bdot_tensor[2,2], bdot_tensor[3,3],
                        bdot_tensor[2,3], bdot_tensor[1,3], bdot_tensor[1,2]]
            ν_storage[i_global,:] = [ν_tensor[1,1], ν_tensor[2,2], ν_tensor[3,3],
                                     ν_tensor[2,3], ν_tensor[1,3], ν_tensor[1,2]]
        else
            β[i_global,:]  = [-2/3*Hkin*b_prev[1,1], -2/3*Hkin*b_prev[2,2], -2/3*Hkin*b_prev[1,2]]
            bdot_vec = [bdot_tensor[1,1], bdot_tensor[2,2], bdot_tensor[1,2]]
            ν_storage[i_global,:] = [ν_tensor[1,1], ν_tensor[2,2], ν_tensor[1,2]]
        end
        κ[i_global] = -abs(Hiso) * k_prev
        bdot[i_global,:] = bdot_vec
        kdot[i_global] = kdot_val

        # NN features (unchanged)
        ν_vec = ν_storage[i_global,:]
        I1[i_global] = κ[i_global]/NN0
        I2[i_global] = dot(β[i_global,:], β[i_global,:])/(NN0^2)
        I3[i_global] = dot(ν_vec, β[i_global,:])/NN0
        I4[i_global] = dot(β[i_global,:], εp[i_global,:])/(NN0^2)
        I5[i_global] = dot(ν_vec, εp[i_global,:])/(NN0^2)
        I6[i_global] = dot(εp[i_global,:], εp[i_global,:])/(NN0^2)

    end
end

# -------------------------
# 8. Add results to DataFrame
for j in 1:ncomp
    df_strain[!, Symbol("εp_"*string(j))] = εp[:,j]
    df_strain[!, Symbol("β_"*string(j))]  = β[:,j]
    df_strain[!, Symbol("b_dot_"*string(j))] = bdot[:,j]
    df_strain[!, Symbol("ν_"*string(j))]  = ν_storage[:,j]
end
df_strain[!,:k_dot] = kdot
df_strain[!,:κ] = κ
df_strain[!,:I1] = I1
df_strain[!,:I2] = I2
df_strain[!,:I3] = I3
df_strain[!,:I4] = I4
df_strain[!,:I5] = I5
df_strain[!,:I6] = I6
df_strain[!,:NNiso]    = NNiso
df_strain[!,:NNk_beta] = NNk_beta
df_strain[!,:NNk_nu]   = NNk_nu

# -------------------------
# 9. Save CSV
CSV.write("strain_with_plastic_NN_inputs.csv", df_strain)

