"""
    sector_dim(dims::Vector{Int}, k::Int=0)

Returns the dimension of the k-labeled diagonal block of the Lindbladian 
of an n-site system with U(1) symmetry. 

Inputs:
- dims[i]: Number of Fockmstates for site i
- k: Integer eigenvalue of the superoperator ±[N, •], where N is the total 
     occupation operator.

The calculation uses the generating function method described in Appendix B 
of https://hdl.handle.net/20.500.11767/124593.
"""
function sector_dim(dims::AbstractVector{<:Integer}, k::Integer=0)
    # Symmetry property: dim(k) = dim(-k)
    k = abs(k)
    
    # The generating function is P(x) = Π_i (Σ_{j=0}^{dims[i]-1} x^j).
    # The coefficients of this polynomial represent the number of states 
    # with a specific total occupation.
    
    # Initialize coefficients with the identity (x^0)
    # BigInt is used to prevent overflow in large many-body systems.
    s_dim = [BigInt(1)]
    
    for cutoff_n in dims
        # A site with 'n' states contributes a polynomial of 'n' terms (all 1s).
        # Example: if cutoff_n = 3, states are |0>, |1>, |2> -> 1 + x + x^2
        poly_site = ones(BigInt, cutoff_n)
        
        # Multiplying polynomials corresponds to the convolution of their coefficients.
        s_dim = simple_conv(s_dim, poly_site)
    end
    
    # The sector dimension is the number of pairs of states with a total 
    # occupation difference equal to k.
    # This is calculated as sum(s_dim[m] * s_dim[m+k]).
    len = length(s_dim)
    if k >= len
        return BigInt(0)
    end
    
    # Use array views for memory efficiency during the final summation.
    v1 = @view s_dim[(1 + k):end]
    v2 = @view s_dim[1:(end - k)]
    
    return sum(v1 .* v2)
end

"""
Helper function to perform 1D convolution for BigInt vectors without external dependencies.
"""
function simple_conv(a::Vector{BigInt}, b::Vector{BigInt})
    n, m = length(a), length(b)
    res = zeros(BigInt, n + m - 1)
    for i in 1:n
        for j in 1:m
            res[i + j - 1] += a[i] * b[j]
        end
    end
    return res
end