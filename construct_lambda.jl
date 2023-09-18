using LinearAlgebra
#using JLD2
using OffsetArrays

LARGE_NUM = 50
alphas_stored = OffsetArray(zeros(LARGE_NUM), -1)
Ts_stored = zeros(LARGE_NUM)
mus_stored = zeros(LARGE_NUM)

function alpha(i)
    if alphas_stored[i] != 0
        return alphas_stored[i]
    end
    val = (-1 * (mu(i)-1) + sqrt((mu(i)-1)^2 + 8 * (beta(i+1)-1)*(mu(i)-1))) / 4 + 1
    alphas_stored[i] = val
    return val
end

function beta(i)
    return 1 + (1 + sqrt(2))^(i-1)
end

function z(k)
    return append!(zeros(2^k-1),sqrt(mu(k)-1), w(k))
end

function w(k)
    res = []
    for j in k:-1:1
        append!(res, [beta(largestpow2(i)) for i in 1:(2^(j-1)-1)]/sqrt(mu(j)-1))
        append!(res, beta(j)/sqrt(mu(j)-1))
    end
    return append!(res, 1)
end

function T(i)
    if Ts_stored[i] != 0
        return Ts_stored[i]
    end
    if i == 1
        val = 0
    else
        val = 2 * sum(Vector{Float64}([alpha(l) for l=0:i-2])) + sum(Vector{Float64}([2 * (2^(i - 1 - l) - 1) * beta(l) for l=0:i-2]))
    end
    Ts_stored[i] = val
    return val
end

function mu(i)
    if i == 0
        return 2
    end
    if mus_stored[i] != 0
        return mus_stored[i]
    end
    val = T(i) + 2 * alpha(i - 1) + 2
    mus_stored[i] = val
    return val
end

# Returns the largest power of 2 dividing n
function largestpow2(n)
    Int(log2(n&-n))
end

function isPowerOf2(n)
    (n & (n-1)) == 0
end

function pi(k)
    temp = zeros(2^k - 1)
    for i = 1:(2^k - 1)
        temp[i] = beta(largestpow2(i))
    end
    return temp
end

function sigma(k)
    if k == 1
        return []
    end
    return append!(
        pi(k - 2),
        beta(k),
        2*(1+sqrt(2))^(2*(k-1)) * sigma(k - 1)
       )/(2*(1+sqrt(2))^(2*(k-1)))
end

function rho(k)
    if k == 0
        return [0,1]
    end
    return append!(
        [(1 + sqrt(2))^(k - 2)],
        ((1 + sqrt(2))^(2 * k - 1) * sigma(k) - pi(k - 1) / (2 * (1 + sqrt(2)))),
        [0.0],
        pi(k),
        [1.0]
    )
end

function pad(k,i,partialRow)
    n = 2^(k+1)+1
    i = i + 1
    l = largestpow2(i)
    if l == 0
        return append!(zeros(i), partialRow, zeros(n - i - 2))
    end
    append!(zeros(i-2^(l-1)), partialRow, zeros(n - (i-2^(l-1)) - length(partialRow)))
end

# function sumi(i)
#     p = count_ones(i+1)
#     z = trunc(Int, log2(i+1))
#     l = largestpow2(i+1)
#     0.5 * (1+sqrt(2))^(2*(p-z-2)) * beta(l + 2) * (mu(z+1)-1)
#     end
# println(maximum(abs.([isPowerOf2(i+1) ? 0 : (sum(ll1023[i:end,i]) - sumi(i)) for i = 1:511])))

function lambdaPart(k,i)
    t = 2^(k+1)-1
    if !isPowerOf2(i+1) && !isPowerOf2(t-i)
        if i +1 < 2^k
            z = trunc(Int, log2(i + 1))
            l = largestpow2(i + 1)
            p = count_ones(i + 1)
            return (mu(z+1)-1) * rho(l) / (1+sqrt(2))^(2*(z-p)+3)
        end 
        z = trunc(Int, log2(t-i))
        l = largestpow2(t-i)
        p = count_ones(i + 1)
        return (1+sqrt(2))^(2*(p+z-k)-1)*rho(l)/(mu(z+1)-1)
    end
    if isPowerOf2(i + 1) && i + 1 < 2^k
        l = largestpow2(i + 1)
        if i == 0
            return [0,2]
        end
        if i == 1
            return [1,0,sqrt(2),1]
        end
        return  append!([(mu(l)-1)/((1+sqrt(2))^(l-1))/2-1],
                       (mu(l)-1)*sigma(l),
                       [0],
                       (mu(l+1)-1)*pi(l)/(2*(1+sqrt(2))^(2*l)),
                       [(mu(l+1)-1)/(2*(1+sqrt(2))^(2*l))]
                      )
    end
    if i == 2^k - 1
        return append!([(mu(k)-1)/((1+sqrt(2))^(k-1))/2-1],
                       (mu(k)-1)*sigma(k),
                       [0],
                       w(k)/sqrt(mu(k)-1)
                      )
    end
    # Last remaining case: i = t - 2^l, for some l
    if i == t - 1
        return [0,0.5]
    end
    if isPowerOf2(t-i)
        l = largestpow2(t-i)
        return append!([(1 + sqrt(2))^(l - 1)]/(mu(l+1)-1),
                        ((1 + sqrt(2))^(2 * l) * sigma(l) - pi(l-1) / (2))/(mu(l+1)-1),
                        [0],
                        w(l)*(1/sqrt(mu(l)-1)-1/sqrt(mu(l+1)-1)))
    end
end

function lambdaRow(k,i)
    t = 2^(k+1)-1
    if i == -1 || i == t
        return zeros(t+2)
    end
    return pad(k,i,lambdaPart(k,i))
    
    return zeros(2^(k+1) +1)
end

function h(k)
    temp = OffsetArray(zeros(2^(k+1) - 1), -1)
    for i = 0:(2^k - 2)
        ind = largestpow2(i+1)
        if (i+1) == Int(2^ind)
            temp[i] = alpha(ind)
        else
            temp[i] = beta(ind)
        end
        temp[2^(k+1) - 2 - i] = temp[i]
    end
    temp[2^k - 1] = 2 * sum(Array{Float64}(temp[0:(2^k - 2)])) + 2
    return temp
end

function lambda(k)
    lambda = zeros(2^(k+1)+1, 2^(k+1)+1)
    for i = -1:(2^(k+1)-1)
        lambda[i+2,:] = lambdaRow(k,i)
    end
    return OffsetArray(lambda,-2,-2)
end

function gvec(i,t)
    # i is index from -1:t
    vec = zeros(t+2)
    if i != -1
        vec[i+2] = 1
    end
    return vec
end

function xvec(i, t, h)
    # i is index from -1:t
    # h is offset array
    vec = zeros(t+2)
    if i == -1
        return vec
    end
    vec[1] = 1
    if i == 0
        return vec
    end
    for j=0:i-1
        vec -= h[j] * gvec(j, t)
    end
    return vec
end

function Amat(i,j,t,h)
    left = gvec(j,t)
    right = xvec(i,t,h) - xvec(j,t,h)
    return (left * right' + right * left') / 2
end

function Cmat(i,j,t)
    left = gvec(i,t) - gvec(j,t)
    right = left
    return (left * right' + right * left') / 2
end

function Mcoeff(i,j,t,h)
    mat = Amat(i,j,t,h) + (1/2) * Cmat(i,j,t)
    mat_ = OffsetArray(mat[2:t+2,2:t+2],-1,-1)
    return mat_
end

function make_M(t,h,λ)
    expression3 = OffsetArray(zeros(t+1, t+1),-1,-1)
    for (i,j) in Iterators.product(-1:t,-1:t)
        if i != j
            expression3 += λ[i,j] * Mcoeff(i,j,t,h)
        end
    end
    return expression3
end

function makeMentry(h, lambda, i, j)
    if i == j
        rowcolsum = sum(lambda[:,i])+sum(lambda[i,:])# + sum(lambda[-1:end,i+1])
        postsum = sum(lambda[i:end,i])
        if postsum == 0
            return rowcolsum/2
        end
        return rowcolsum/2 - h[i] * postsum
    end    
    if i < j
        j,i = i,j
    end
    postsum1 = sum(lambda[i+1:end,j])
    if postsum1 != 0
      postsum1 = h[i] * postsum1 / 2
    end
    postsum2 = sum(lambda[-1:j,i])
    if postsum2 != 0
      postsum2 =  h[j] * postsum2 / 2 
    end
    return -(lambda[i,j] + lambda[j,i])/2 - postsum1 + postsum2
end