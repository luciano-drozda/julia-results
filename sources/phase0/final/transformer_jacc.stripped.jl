using CUDA
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_transformer_1!(__jacc_i, b_offset, bq, d, n_d, q, w_offset, wq, x)
    idx = 1 + (__jacc_i - 1)
    i = div(idx - 1, d) + 1
    j = mod(idx - 1, d) + 1
    s = 0.0
    for i_p = 1:d
        s = s + x[(i - 1) * d + i_p] * wq[w_offset + (i_p - 1) * d + j]
    end
    q[(i - 1) * d + j] = s + bq[b_offset + j]
    return nothing
end

function jacc_kernel_transformer_2!(__jacc_i, b_offset, bk, d, k, n_d, w_offset, wk, x)
    idx = 1 + (__jacc_i - 1)
    i = div(idx - 1, d) + 1
    j = mod(idx - 1, d) + 1
    s = 0.0
    for i_p = 1:d
        s = s + x[(i - 1) * d + i_p] * wk[w_offset + (i_p - 1) * d + j]
    end
    k[(i - 1) * d + j] = s + bk[b_offset + j]
    return nothing
end

function jacc_kernel_transformer_3!(__jacc_i, b_offset, bv, d, n_d, v, w_offset, wv, x)
    idx = 1 + (__jacc_i - 1)
    i = div(idx - 1, d) + 1
    j = mod(idx - 1, d) + 1
    s = 0.0
    for i_p = 1:d
        s = s + x[(i - 1) * d + i_p] * wv[w_offset + (i_p - 1) * d + j]
    end
    v[(i - 1) * d + j] = s + bv[b_offset + j]
    return nothing
end

function jacc_kernel_transformer_4!(__jacc_i, d, dk, head_offset, inv_sqrt_dk, k, n, q, score_off, scores)
    idx2 = 1 + (__jacc_i - 1)
    i = div(idx2 - 1, n) + 1
    j = mod(idx2 - 1, n) + 1
    s = 0.0
    for i_p = 1:dk
        s = s + q[(i - 1) * d + head_offset + i_p] * k[(j - 1) * d + head_offset + i_p]
    end
    scores[score_off + (i - 1) * n + j] = s * inv_sqrt_dk
    return nothing
end

function jacc_kernel_transformer_5!(__jacc_i, i, n, probs, row_max, score_off, scores)
    j = 1 + (__jacc_i - 1)
    kk = score_off + (i - 1) * n + j
    probs[kk] = exp(scores[kk] - row_max)
    return nothing
end

function jacc_kernel_transformer_6!(__jacc_i, i, n, probs, row_sum, score_off)
    j = 1 + (__jacc_i - 1)
    kk = score_off + (i - 1) * n + j
    probs[kk] = probs[kk] / row_sum
    return nothing
end

function jacc_kernel_transformer_7!(__jacc_i, ctx, d, dk, head_offset, n, probs, score_off, v)
    idx3 = 1 + (__jacc_i - 1)
    i = div(idx3 - 1, dk) + 1
    p = mod(idx3 - 1, dk) + 1
    s = 0.0
    for i_j = 1:n
        s = s + probs[score_off + (i - 1) * n + i_j] * v[(i_j - 1) * d + head_offset + p]
    end
    ctx[(i - 1) * d + head_offset + p] = s
    return nothing
end

function jacc_kernel_transformer_8!(__jacc_i, attn_out, b_offset, bo, ctx, d, n_d, w_offset, wo)
    idx = 1 + (__jacc_i - 1)
    i = div(idx - 1, d) + 1
    j = mod(idx - 1, d) + 1
    s = 0.0
    for i_p = 1:d
        s = s + ctx[(i - 1) * d + i_p] * wo[w_offset + (i_p - 1) * d + j]
    end
    attn_out[(i - 1) * d + j] = s + bo[b_offset + j]
    return nothing
end

function jacc_kernel_transformer_9!(__jacc_i, attn_out, n_d, resid1, x)
    idx = 1 + (__jacc_i - 1)
    resid1[idx] = x[idx] + attn_out[idx]
    return nothing
end

function jacc_kernel_transformer_10!(__jacc_i, d, eps, ln1_bias, ln1_gain, ln_offset, n, normed1, resid1)
    i = 1 + (__jacc_i - 1)
    s = 0.0
    for i_j = 1:d
        s = s + resid1[(i - 1) * d + i_j]
    end
    row_mean = s / d
    s2 = 0.0
    for i_j = 1:d
        diff = resid1[(i - 1) * d + i_j] - row_mean
        s2 = s2 + diff * diff
    end
    row_var = s2 / d
    denom = sqrt(row_var + eps)
    for j = 1:d
        kk = (i - 1) * d + j
        normed1[kk] = ((resid1[kk] - row_mean) / denom) * ln1_gain[ln_offset + j] + ln1_bias[ln_offset + j]
    end
    return nothing
end

function jacc_kernel_transformer_11!(__jacc_i, b1, b1_offset, d, dff, ff_hidden, n_dff, normed1, w1, w1_offset)
    idx = 1 + (__jacc_i - 1)
    i = div(idx - 1, dff) + 1
    j = mod(idx - 1, dff) + 1
    s = 0.0
    for i_p = 1:d
        s = s + normed1[(i - 1) * d + i_p] * w1[w1_offset + (i_p - 1) * dff + j]
    end
    ff_hidden[(i - 1) * dff + j] = max(s + b1[b1_offset + j], 0.0)
    return nothing
end

function jacc_kernel_transformer_12!(__jacc_i, b2, b2_offset, d, dff, ff_hidden, ff_out, n_d, w2, w2_offset)
    idx = 1 + (__jacc_i - 1)
    i = div(idx - 1, d) + 1
    j = mod(idx - 1, d) + 1
    s = 0.0
    for i_p = 1:dff
        s = s + ff_hidden[(i - 1) * dff + i_p] * w2[w2_offset + (i_p - 1) * d + j]
    end
    ff_out[(i - 1) * d + j] = s + b2[b2_offset + j]
    return nothing
end

function jacc_kernel_transformer_13!(__jacc_i, ff_out, n_d, normed1, resid2)
    idx = 1 + (__jacc_i - 1)
    resid2[idx] = normed1[idx] + ff_out[idx]
    return nothing
end

function jacc_kernel_transformer_14!(__jacc_i, d, eps, ln2_bias, ln2_gain, ln2_offset, n, resid2, x_next)
    i = 1 + (__jacc_i - 1)
    s = 0.0
    for i_j = 1:d
        s = s + resid2[(i - 1) * d + i_j]
    end
    row_mean = s / d
    s2 = 0.0
    for i_j = 1:d
        diff = resid2[(i - 1) * d + i_j] - row_mean
        s2 = s2 + diff * diff
    end
    row_var = s2 / d
    denom = sqrt(row_var + eps)
    for j = 1:d
        kk = (i - 1) * d + j
        x_next[kk] = ((resid2[kk] - row_mean) / denom) * ln2_gain[ln2_offset + j] + ln2_bias[ln2_offset + j]
    end
    return nothing
end

function jacc_kernel_transformer_15!(__jacc_i, n_d, x, x_next)
    idx = 1 + (__jacc_i - 1)
    x[idx] = x_next[idx]
    return nothing
end

function transformer_jacc(x, wq, bq, wk, bk, wv, bv, wo, bo, ln1_gain, ln1_bias, w1, b1, w2, b2, ln2_gain, ln2_bias, q, k, v, scores, probs, ctx, attn_out, resid1, normed1, ff_hidden, ff_out, resid2, x_next, n, dk, h, dff, n_layers, eps)
    d = h * dk
    inv_sqrt_dk = 1.0 / sqrt(dk)
    n_d = n * d
    n_dff = n * dff
    for i_l = 1:n_layers
        w_offset = (i_l - 1) * d * d
        b_offset = (i_l - 1) * d
        ln_offset = (i_l - 1) * d
        w1_offset = (i_l - 1) * d * dff
        b1_offset = (i_l - 1) * dff
        w2_offset = (i_l - 1) * dff * d
        b2_offset = (i_l - 1) * d
        ln2_offset = (i_l - 1) * d
        if div(n_d - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d - 1, 1) + 1 jacc_kernel_transformer_1!(b_offset, bq, d, n_d, q, w_offset, wq, x)
        end
        if div(n_d - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d - 1, 1) + 1 jacc_kernel_transformer_2!(b_offset, bk, d, k, n_d, w_offset, wk, x)
        end
        if div(n_d - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d - 1, 1) + 1 jacc_kernel_transformer_3!(b_offset, bv, d, n_d, v, w_offset, wv, x)
        end
        for hh = 1:h
            head_offset = (hh - 1) * dk
            score_off = (hh - 1) * n * n
            if div(n * n - 1, 1) + 1 > 0
                JACC.@parallel_for range = div(n * n - 1, 1) + 1 jacc_kernel_transformer_4!(d, dk, head_offset, inv_sqrt_dk, k, n, q, score_off, scores)
            end
            for i = 1:n
                CUDA.@allowscalar begin
                        row_max = scores[score_off + (i - 1) * n + 1]
                    end
                for i_j = 2:n
                    CUDA.@allowscalar begin
                            row_max = max(row_max, scores[score_off + (i - 1) * n + i_j])
                        end
                end
                if div(n - 1, 1) + 1 > 0
                    JACC.@parallel_for range = div(n - 1, 1) + 1 jacc_kernel_transformer_5!(i, n, probs, row_max, score_off, scores)
                end
                row_sum = 0.0
                for i_j = 1:n
                    CUDA.@allowscalar begin
                            row_sum = row_sum + probs[score_off + (i - 1) * n + i_j]
                        end
                end
                if div(n - 1, 1) + 1 > 0
                    JACC.@parallel_for range = div(n - 1, 1) + 1 jacc_kernel_transformer_6!(i, n, probs, row_sum, score_off)
                end
            end
            if div(n * dk - 1, 1) + 1 > 0
                JACC.@parallel_for range = div(n * dk - 1, 1) + 1 jacc_kernel_transformer_7!(ctx, d, dk, head_offset, n, probs, score_off, v)
            end
        end
        if div(n_d - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d - 1, 1) + 1 jacc_kernel_transformer_8!(attn_out, b_offset, bo, ctx, d, n_d, w_offset, wo)
        end
        if div(n_d - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d - 1, 1) + 1 jacc_kernel_transformer_9!(attn_out, n_d, resid1, x)
        end
        if div(n - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n - 1, 1) + 1 jacc_kernel_transformer_10!(d, eps, ln1_bias, ln1_gain, ln_offset, n, normed1, resid1)
        end
        if div(n_dff - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_dff - 1, 1) + 1 jacc_kernel_transformer_11!(b1, b1_offset, d, dff, ff_hidden, n_dff, normed1, w1, w1_offset)
        end
        if div(n_d - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d - 1, 1) + 1 jacc_kernel_transformer_12!(b2, b2_offset, d, dff, ff_hidden, ff_out, n_d, w2, w2_offset)
        end
        if div(n_d - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d - 1, 1) + 1 jacc_kernel_transformer_13!(ff_out, n_d, normed1, resid2)
        end
        if div(n - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n - 1, 1) + 1 jacc_kernel_transformer_14!(d, eps, ln2_bias, ln2_gain, ln2_offset, n, resid2, x_next)
        end
        if div(n_d - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d - 1, 1) + 1 jacc_kernel_transformer_15!(n_d, x, x_next)
        end
    end
    return nothing
end
