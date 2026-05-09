using QuantumToolbox
"""
write the basis in number representation, 2 sites with dims 3 (meaning each site can be in state 0,1,2)
"""
dims = (3, 3)

ket_basis = vec(collect(Iterators.product(0:dims[1]-1, 0:dims[2]-1)))
#creating a dictionary
ket2idx = Dict(s => i for (i,s) in enumerate(ket_basis))
idx2ket = Dict(i => s for (i,s) in enumerate(ket_basis))
#=
for (i, s) in idx2ket
    println("Dictionary: Index: $i, Ket: $s")
end
for k in ket_basis
    println("Ket basis: $k")
end
=#
n_exc = 1  # max |q|

super_basis = []
for (i, ket) in enumerate(ket_basis)
    for (j, bra) in enumerate(ket_basis)
        q = sum(ket) - sum(bra)
        if abs(q) <= n_exc
            push!(super_basis, (ket, bra, q))
        end
    end
end

# assign indices
super2idx = Dict((s[1], s[2]) => i for (i,s) in enumerate(super_basis))
idx2super = Dict(i => s for (i,s) in enumerate(super_basis))
println(super_basis)
println("Full super space: $(length(ket_basis)^2) states")
println("Restricted super space: $(length(super_basis)) states")
#step 3 is to build operators in the restricted super space
