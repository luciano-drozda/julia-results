# red_rev3: sum of squares
#
# u: input array
# loss: length-1 output array, accumulated in place
function red_rev3(u, loss, i_n)
    for i_x = i_n:-1:3
        loss[1] = loss[1] + u[i_x] ^ 2
    end
    return nothing
end
