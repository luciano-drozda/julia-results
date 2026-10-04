# red_lo3s3: sum of squares
#
# u: input array
# loss: length-1 output array, accumulated in place
function red_lo3s3(u, loss, i_n)
    for i_x = 3:3:i_n
        loss[1] = loss[1] + u[i_x] ^ 2
    end
    return nothing
end
