using QuantumToolbox
"""
I defined my self made restricted super space with number representation, need to understand the mapping
between the states in the super space and the states in the full hilbert space
indexing rule: if we have dims = (d1,d2,...,dL) and ket = |n1,n2,...,nL>, then 
ket_index = n1(d2*d3*...*dL) + n2(d3*d4*...*dL) + ... + nL-1(dL) + nL + 1 
"""


ψ = basis(3,2) ⊗ basis(3,0) 
ψ2 = basis(3,1) ⊗ basis(3,2)

println("ψ = $ψ")
println("ψ2 = $ψ2")
"""
indexing for density matrices: rho_{ij} with i ket index and j bra index 
superspace indices, the formula is: 
    super_index = (bra_index - 1) * dim_ket + ket_index
    where dim_ket is the dimension of the ket space.
"""
ρ = ψ * ψ2'
s_index = findfirst(==(1), vec_ρ.data)
println("superindex = $s_index")


