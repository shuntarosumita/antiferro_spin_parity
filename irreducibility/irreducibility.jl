struct Parameters
    twoS::Int
    L::Int
end

"an operator for adding -1 and +1 to j1-th and j2-th numbers, respectively"
function operate_R(p::Parameters, n_in::AbstractVector, j1::Int, j2::Int)
    ## error check
    (1 <= j1 <= p.L) || error("range error: j1=", j1, ", L=", p.L)
    (1 <= j2 <= p.L) || error("range error: j2=", j2, ", L=", p.L)
    (j1 != j2) || error("error: j1 == j2")
    (n_in[j1] > 0 && n_in[j2] < p.twoS) || error("error: n[j1] = 0 or n[j2] = 2S")

    ## set indices
    Λ::Vector{Int} = Vector(min(j1, j2)+1:max(j1, j2)-1)
    Λ_0::Vector{Int} = Λ[n_in[Λ] .== 0]

    n_out = copy(n_in)

    ## (1) add 2 to all zero numbers
    for j in Λ_0
        n_out[j] += 2
        println("Q_$j^(+):\t $n_out")
    end

    ## (2) transfer -1 from j2 to j1
    if j1 < j2
        for j in j2-1:-1:j1
            n_out[j] -= 1; n_out[mod(j+1, 1:p.L)] += 1
            println("P_$j^(-+):\t $n_out")
        end
    else
        for j in j2:j1-1
            n_out[j] += 1; n_out[mod(j+1, 1:p.L)] -= 1
            println("P_$j^(+-):\t $n_out")
        end
    end

    ## (3) reduce 2 at the positions raised in (1)
    for j in Λ_0
        n_out[j] -= 2
        println("Q_$j^(-):\t $n_out")
    end

    n_out
end

function main()
    ## input from command line arguments
    length(ARGS) < 3 && error("usage: julia main.jl twoS n1 n2")
    twoS::Int = parse(Int, ARGS[1])
    n1::Vector{Int} = parse.(Int, collect(ARGS[2]))
    n2::Vector{Int} = parse.(Int, collect(ARGS[3]))

    ## error check
    (2 <= twoS <= 9) || error("S does not satisfy 2 <= 2S <= 9")
    length(n1) == length(n2) || error("dimension mismatch: length(n1) != length(n2)")
    (maximum(n1) <= twoS && maximum(n2) <= twoS) || error("maximum exceeds 2S")
    sum(n2 .- n1) % 2 == 0 || error("parity mismatch: sum(n1 - n2) is odd")

    ## main part
    p::Parameters = Parameters(twoS, length(n1))
    Δn::Vector{Int} = n2 .- n1
    Δn_tot::Int = sum(Δn)
    println("initial state:\t $n1")

    ### iteration to make Δn_tot zero
    while Δn_tot != 0
        if Δn_tot < 0
            Λ_max = findall(n -> n == maximum(n1), n1)
            if n1[Λ_max[1]] >= 2
                n1[Λ_max[1]] -= 2
                println("Q_$(Λ_max[1])^(-):\t $n1")
            else
                n1 .= operate_R(p, n1, Λ_max[2], Λ_max[1])
            end
        else
            Λ_min = findall(n -> n == minimum(n1), n1)
            if n1[Λ_min[1]] <= twoS - 2
                n1[Λ_min[1]] += 2
                println("Q_$(Λ_min[1])^(+):\t $n1")
            else
                n1 .= operate_R(p, n1, Λ_min[1], Λ_min[2])
            end
        end

        Δn = n2 .- n1
        Δn_tot = sum(Δn)
    end

    ### iteration to make Δn zero
    while !all(iszero, Δn)
        n1 .= operate_R(p, n1, argmin(Δn), argmax(Δn))
        Δn .= n2 .- n1
    end

    nothing
end

main()