using CUDA
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_transformer_loss_b_1!(__jacc_i, dk, h, n, x, x_in)
    i_c = 1 + (__jacc_i - 1)
    x[i_c] = x_in[i_c]
    return nothing
end

function jacc_kernel_transformer_loss_b_2!(__jacc_i, b_offset_transformer_c1, bq, d_transformer_c1, i_l_transformer_c1, n_d_transformer_c1, q, q_stack, s_transformer_c1_stack, w_offset_transformer_c1, wq, x)
    idx_transformer_c1 = 1 + (__jacc_i - 1)
    __icse_2 = idx_transformer_c1 - 1
    i_transformer_c1 = div(__icse_2, d_transformer_c1) + 1
    j_transformer_c1 = mod(__icse_2, d_transformer_c1) + 1
    s_transformer_c1 = 0.0
    for i_p_transformer_c1 = 1:d_transformer_c1
        s_transformer_c1 = s_transformer_c1 + x[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1] * wq[w_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
    end
    __icse_3 = ((i_l_transformer_c1 - 1) * (div(n_d_transformer_c1 - 1, 1) + 1) + (idx_transformer_c1 - 1)) + 1
    __idx_q_stack_4 = __icse_3
    q_stack[__idx_q_stack_4] = q[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
    q[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] = s_transformer_c1 + bq[b_offset_transformer_c1 + j_transformer_c1]
    __idx_s_transformer_c1_stack_7 = __icse_3
    s_transformer_c1_stack[__idx_s_transformer_c1_stack_7] = s_transformer_c1
    return nothing
end

function jacc_kernel_transformer_loss_b_3!(__jacc_i, b_offset_transformer_c1, bk, d_transformer_c1, i_l_transformer_c1, k, k_stack, n_d_transformer_c1, n_layers, s_transformer_c1_stack, w_offset_transformer_c1, wk, x)
    idx_transformer_c1 = 1 + (__jacc_i - 1)
    __icse_4 = idx_transformer_c1 - 1
    i_transformer_c1 = div(__icse_4, d_transformer_c1) + 1
    j_transformer_c1 = mod(__icse_4, d_transformer_c1) + 1
    s_transformer_c1 = 0.0
    for i_p_transformer_c1 = 1:d_transformer_c1
        s_transformer_c1 = s_transformer_c1 + x[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1] * wk[w_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
    end
    __icse_5 = div(n_d_transformer_c1 - 1, 1) + 1
    __icse_6 = ((i_l_transformer_c1 - 1) * __icse_5 + (idx_transformer_c1 - 1)) + 1
    __idx_k_stack_4 = __icse_6
    k_stack[__idx_k_stack_4] = k[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
    k[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] = s_transformer_c1 + bk[b_offset_transformer_c1 + j_transformer_c1]
    __idx_s_transformer_c1_stack_7 = max(0, div(n_layers - 1, 1) + 1) * max(0, __icse_5) + __icse_6
    s_transformer_c1_stack[__idx_s_transformer_c1_stack_7] = s_transformer_c1
    return nothing
end

function jacc_kernel_transformer_loss_b_4!(__jacc_i, b_offset_transformer_c1, bv, d_transformer_c1, i_l_transformer_c1, n_d_transformer_c1, n_layers, s_transformer_c1_stack, v, v_stack, w_offset_transformer_c1, wv, x)
    idx_transformer_c1 = 1 + (__jacc_i - 1)
    __icse_7 = idx_transformer_c1 - 1
    i_transformer_c1 = div(__icse_7, d_transformer_c1) + 1
    j_transformer_c1 = mod(__icse_7, d_transformer_c1) + 1
    s_transformer_c1 = 0.0
    for i_p_transformer_c1 = 1:d_transformer_c1
        s_transformer_c1 = s_transformer_c1 + x[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1] * wv[w_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
    end
    __icse_8 = div(n_d_transformer_c1 - 1, 1) + 1
    __icse_9 = ((i_l_transformer_c1 - 1) * __icse_8 + (idx_transformer_c1 - 1)) + 1
    __idx_v_stack_4 = __icse_9
    v_stack[__idx_v_stack_4] = v[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
    v[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] = s_transformer_c1 + bv[b_offset_transformer_c1 + j_transformer_c1]
    __icse_10 = max(0, div(n_layers - 1, 1) + 1) * max(0, __icse_8)
    __idx_s_transformer_c1_stack_7 = (__icse_10 + __icse_10) + __icse_9
    s_transformer_c1_stack[__idx_s_transformer_c1_stack_7] = s_transformer_c1
    return nothing
end

function jacc_kernel_transformer_loss_b_5!(__jacc_i, d_transformer_c1, dk, h, head_offset_transformer_c1, hh_transformer_c1, i_l_transformer_c1, inv_sqrt_dk_transformer_c1, k, n, n_d_transformer_c1, n_layers, q, s_transformer_c1_stack, score_off_transformer_c1, scores, scores_stack)
    idx2_transformer_c1 = 1 + (__jacc_i - 1)
    __icse_12 = idx2_transformer_c1 - 1
    i_transformer_c1 = div(__icse_12, n) + 1
    j_transformer_c1 = mod(__icse_12, n) + 1
    s_transformer_c1 = 0.0
    for i_p_transformer_c1 = 1:dk
        s_transformer_c1 = s_transformer_c1 + q[(i_transformer_c1 - 1) * d_transformer_c1 + head_offset_transformer_c1 + i_p_transformer_c1] * k[(j_transformer_c1 - 1) * d_transformer_c1 + head_offset_transformer_c1 + i_p_transformer_c1]
    end
    __icse_13 = div(n * n - 1, 1) + 1
    __icse_14 = ((i_l_transformer_c1 - 1) * ((div(h - 1, 1) + 1) * __icse_13) + (hh_transformer_c1 - 1) * __icse_13 + (idx2_transformer_c1 - 1)) + 1
    __idx_scores_stack_4 = __icse_14
    scores_stack[__idx_scores_stack_4] = scores[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + j_transformer_c1]
    scores[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + j_transformer_c1] = s_transformer_c1 * inv_sqrt_dk_transformer_c1
    __icse_15 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d_transformer_c1 - 1, 1) + 1)
    __idx_s_transformer_c1_stack_7 = ((__icse_15 + __icse_15) + __icse_15) + __icse_14
    s_transformer_c1_stack[__idx_s_transformer_c1_stack_7] = s_transformer_c1
    return nothing
end

function jacc_kernel_transformer_loss_b_6!(__jacc_i, h, hh_transformer_c1, i_l_transformer_c1, i_transformer_c1, n, probs, probs_stack, row_max_transformer_c1, score_off_transformer_c1, scores)
    j_transformer_c1 = 1 + (__jacc_i - 1)
    __icse_18 = i_transformer_c1 - 1
    kk_transformer_c1 = score_off_transformer_c1 + __icse_18 * n + j_transformer_c1
    __icse_19 = div(n - 1, 1) + 1
    __idx_probs_stack_1 = ((i_l_transformer_c1 - 1) * ((div(h - 1, 1) + 1) * __icse_19 * __icse_19) + (hh_transformer_c1 - 1) * (__icse_19 * __icse_19) + __icse_18 * __icse_19 + (j_transformer_c1 - 1)) + 1
    probs_stack[__idx_probs_stack_1] = probs[kk_transformer_c1]
    probs[kk_transformer_c1] = exp(scores[kk_transformer_c1] - row_max_transformer_c1)
    return nothing
end

function jacc_kernel_transformer_loss_b_7!(__jacc_i, h, hh_transformer_c1, i_l_transformer_c1, i_transformer_c1, n, n_layers, probs, probs_stack, row_sum_transformer_c1, score_off_transformer_c1)
    j_transformer_c1 = 1 + (__jacc_i - 1)
    __icse_20 = i_transformer_c1 - 1
    kk_transformer_c1 = score_off_transformer_c1 + __icse_20 * n + j_transformer_c1
    __icse_21 = div(h - 1, 1) + 1
    __icse_22 = div(n - 1, 1) + 1
    __icse_23 = max(0, __icse_22)
    __idx_probs_stack_1 = max(0, div(n_layers - 1, 1) + 1) * max(0, __icse_21) * __icse_23 * __icse_23 + (((i_l_transformer_c1 - 1) * (__icse_21 * __icse_22 * __icse_22) + (hh_transformer_c1 - 1) * (__icse_22 * __icse_22) + __icse_20 * __icse_22 + (j_transformer_c1 - 1)) + 1)
    __cse_24 = probs[kk_transformer_c1]
    probs_stack[__idx_probs_stack_1] = __cse_24
    probs[kk_transformer_c1] = __cse_24 / row_sum_transformer_c1
    return nothing
end

function jacc_kernel_transformer_loss_b_8!(__jacc_i, ctx, ctx_stack, d_transformer_c1, dk, h, head_offset_transformer_c1, hh_transformer_c1, i_l_transformer_c1, n, n_d_transformer_c1, n_layers, probs, s_transformer_c1_stack, score_off_transformer_c1, v)
    idx3_transformer_c1 = 1 + (__jacc_i - 1)
    __icse_28 = idx3_transformer_c1 - 1
    i_transformer_c1 = div(__icse_28, dk) + 1
    p_transformer_c1 = mod(__icse_28, dk) + 1
    s_transformer_c1 = 0.0
    for i_j_transformer_c1 = 1:n
        s_transformer_c1 = s_transformer_c1 + probs[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + i_j_transformer_c1] * v[(i_j_transformer_c1 - 1) * d_transformer_c1 + head_offset_transformer_c1 + p_transformer_c1]
    end
    __icse_29 = div(h - 1, 1) + 1
    __icse_30 = div(n * dk - 1, 1) + 1
    __icse_31 = ((i_l_transformer_c1 - 1) * (__icse_29 * __icse_30) + (hh_transformer_c1 - 1) * __icse_30 + (idx3_transformer_c1 - 1)) + 1
    __idx_ctx_stack_4 = __icse_31
    ctx_stack[__idx_ctx_stack_4] = ctx[(i_transformer_c1 - 1) * d_transformer_c1 + head_offset_transformer_c1 + p_transformer_c1]
    ctx[(i_transformer_c1 - 1) * d_transformer_c1 + head_offset_transformer_c1 + p_transformer_c1] = s_transformer_c1
    __icse_32 = max(0, div(n_layers - 1, 1) + 1)
    __icse_33 = __icse_32 * max(0, div(n_d_transformer_c1 - 1, 1) + 1)
    __idx_s_transformer_c1_stack_7 = (((__icse_33 + __icse_33) + __icse_33) + __icse_32 * max(0, __icse_29) * max(0, div(n * n - 1, 1) + 1)) + __icse_31
    s_transformer_c1_stack[__idx_s_transformer_c1_stack_7] = s_transformer_c1
    return nothing
end

function jacc_kernel_transformer_loss_b_9!(__jacc_i, attn_out, b_offset_transformer_c1, bo, ctx, d_transformer_c1, dk, h, i_l_transformer_c1, n, n_d_transformer_c1, n_layers, s_transformer_c1_stack, w_offset_transformer_c1, wo)
    idx_transformer_c1 = 1 + (__jacc_i - 1)
    __icse_34 = idx_transformer_c1 - 1
    i_transformer_c1 = div(__icse_34, d_transformer_c1) + 1
    j_transformer_c1 = mod(__icse_34, d_transformer_c1) + 1
    s_transformer_c1 = 0.0
    for i_p_transformer_c1 = 1:d_transformer_c1
        s_transformer_c1 = s_transformer_c1 + ctx[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1] * wo[w_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
    end
    attn_out[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] = s_transformer_c1 + bo[b_offset_transformer_c1 + j_transformer_c1]
    __icse_35 = max(0, div(n_layers - 1, 1) + 1)
    __icse_36 = div(n_d_transformer_c1 - 1, 1) + 1
    __icse_37 = __icse_35 * max(0, __icse_36)
    __icse_38 = max(0, div(h - 1, 1) + 1)
    __idx_s_transformer_c1_stack_5 = ((((__icse_37 + __icse_37) + __icse_37) + __icse_35 * __icse_38 * max(0, div(n * n - 1, 1) + 1)) + __icse_35 * __icse_38 * max(0, div(n * dk - 1, 1) + 1)) + (((i_l_transformer_c1 - 1) * __icse_36 + (idx_transformer_c1 - 1)) + 1)
    s_transformer_c1_stack[__idx_s_transformer_c1_stack_5] = s_transformer_c1
    return nothing
end

function jacc_kernel_transformer_loss_b_10!(__jacc_i, attn_out, i_l_transformer_c1, n_d_transformer_c1, resid1, resid1_stack, x)
    idx_transformer_c1 = 1 + (__jacc_i - 1)
    __idx_resid1_stack_0 = ((i_l_transformer_c1 - 1) * (div(n_d_transformer_c1 - 1, 1) + 1) + (idx_transformer_c1 - 1)) + 1
    resid1_stack[__idx_resid1_stack_0] = resid1[idx_transformer_c1]
    resid1[idx_transformer_c1] = x[idx_transformer_c1] + attn_out[idx_transformer_c1]
    return nothing
end

function jacc_kernel_transformer_loss_b_11!(__jacc_i, d_transformer_c1, denom_transformer_c1, i_l_transformer_c1, i_transformer_c1, ln1_bias, ln1_gain, ln_offset_transformer_c1, n, normed1, normed1_stack, resid1, row_mean_transformer_c1)
    j_transformer_c1 = 1 + (__jacc_i - 1)
    __icse_40 = i_transformer_c1 - 1
    kk_transformer_c1 = __icse_40 * d_transformer_c1 + j_transformer_c1
    __icse_41 = div(d_transformer_c1 - 1, 1) + 1
    __idx_normed1_stack_1 = ((i_l_transformer_c1 - 1) * ((div(n - 1, 1) + 1) * __icse_41) + __icse_40 * __icse_41 + (j_transformer_c1 - 1)) + 1
    normed1_stack[__idx_normed1_stack_1] = normed1[kk_transformer_c1]
    normed1[kk_transformer_c1] = ((resid1[kk_transformer_c1] - row_mean_transformer_c1) / denom_transformer_c1) * ln1_gain[ln_offset_transformer_c1 + j_transformer_c1] + ln1_bias[ln_offset_transformer_c1 + j_transformer_c1]
    return nothing
end

function jacc_kernel_transformer_loss_b_12!(__jacc_i, b1, b1_offset_transformer_c1, d_transformer_c1, dff, dk, ff_hidden, ff_hidden_stack, h, i_l_transformer_c1, n, n_d_transformer_c1, n_dff_transformer_c1, n_layers, normed1, s_transformer_c1_stack, w1, w1_offset_transformer_c1)
    idx_transformer_c1 = 1 + (__jacc_i - 1)
    __icse_46 = idx_transformer_c1 - 1
    i_transformer_c1 = div(__icse_46, dff) + 1
    j_transformer_c1 = mod(__icse_46, dff) + 1
    s_transformer_c1 = 0.0
    for i_p_transformer_c1 = 1:d_transformer_c1
        s_transformer_c1 = s_transformer_c1 + normed1[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1] * w1[w1_offset_transformer_c1 + (i_p_transformer_c1 - 1) * dff + j_transformer_c1]
    end
    __icse_47 = ((i_l_transformer_c1 - 1) * (div(n_dff_transformer_c1 - 1, 1) + 1) + (idx_transformer_c1 - 1)) + 1
    __idx_ff_hidden_stack_4 = __icse_47
    ff_hidden_stack[__idx_ff_hidden_stack_4] = ff_hidden[(i_transformer_c1 - 1) * dff + j_transformer_c1]
    ff_hidden[(i_transformer_c1 - 1) * dff + j_transformer_c1] = max(s_transformer_c1 + b1[b1_offset_transformer_c1 + j_transformer_c1], 0.0)
    __icse_48 = max(0, div(n_layers - 1, 1) + 1)
    __icse_49 = __icse_48 * max(0, div(n_d_transformer_c1 - 1, 1) + 1)
    __icse_50 = max(0, div(h - 1, 1) + 1)
    __idx_s_transformer_c1_stack_7 = ((((((__icse_49 + __icse_49) + __icse_49) + __icse_48 * __icse_50 * max(0, div(n * n - 1, 1) + 1)) + __icse_48 * __icse_50 * max(0, div(n * dk - 1, 1) + 1)) + __icse_49) + __icse_48 * max(0, div(n - 1, 1) + 1)) + __icse_47
    s_transformer_c1_stack[__idx_s_transformer_c1_stack_7] = s_transformer_c1
    return nothing
end

function jacc_kernel_transformer_loss_b_13!(__jacc_i, b2, b2_offset_transformer_c1, d_transformer_c1, dff, dk, ff_hidden, ff_out, h, i_l_transformer_c1, n, n_d_transformer_c1, n_dff_transformer_c1, n_layers, s_transformer_c1_stack, w2, w2_offset_transformer_c1)
    idx_transformer_c1 = 1 + (__jacc_i - 1)
    __icse_51 = idx_transformer_c1 - 1
    i_transformer_c1 = div(__icse_51, d_transformer_c1) + 1
    j_transformer_c1 = mod(__icse_51, d_transformer_c1) + 1
    s_transformer_c1 = 0.0
    for i_p_transformer_c1 = 1:dff
        s_transformer_c1 = s_transformer_c1 + ff_hidden[(i_transformer_c1 - 1) * dff + i_p_transformer_c1] * w2[w2_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
    end
    ff_out[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] = s_transformer_c1 + b2[b2_offset_transformer_c1 + j_transformer_c1]
    __icse_52 = max(0, div(n_layers - 1, 1) + 1)
    __icse_53 = div(n_d_transformer_c1 - 1, 1) + 1
    __icse_54 = __icse_52 * max(0, __icse_53)
    __icse_55 = max(0, div(h - 1, 1) + 1)
    __idx_s_transformer_c1_stack_5 = (((((((__icse_54 + __icse_54) + __icse_54) + __icse_52 * __icse_55 * max(0, div(n * n - 1, 1) + 1)) + __icse_52 * __icse_55 * max(0, div(n * dk - 1, 1) + 1)) + __icse_54) + __icse_52 * max(0, div(n - 1, 1) + 1)) + __icse_52 * max(0, div(n_dff_transformer_c1 - 1, 1) + 1)) + (((i_l_transformer_c1 - 1) * __icse_53 + (idx_transformer_c1 - 1)) + 1)
    s_transformer_c1_stack[__idx_s_transformer_c1_stack_5] = s_transformer_c1
    return nothing
end

function jacc_kernel_transformer_loss_b_14!(__jacc_i, ff_out, i_l_transformer_c1, n_d_transformer_c1, normed1, resid2, resid2_stack)
    idx_transformer_c1 = 1 + (__jacc_i - 1)
    __idx_resid2_stack_0 = ((i_l_transformer_c1 - 1) * (div(n_d_transformer_c1 - 1, 1) + 1) + (idx_transformer_c1 - 1)) + 1
    resid2_stack[__idx_resid2_stack_0] = resid2[idx_transformer_c1]
    resid2[idx_transformer_c1] = normed1[idx_transformer_c1] + ff_out[idx_transformer_c1]
    return nothing
end

function jacc_kernel_transformer_loss_b_15!(__jacc_i, d_transformer_c1, denom_transformer_c1, i_transformer_c1, ln2_bias, ln2_gain, ln2_offset_transformer_c1, resid2, row_mean_transformer_c1, x_next)
    j_transformer_c1 = 1 + (__jacc_i - 1)
    kk_transformer_c1 = (i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1
    x_next[kk_transformer_c1] = ((resid2[kk_transformer_c1] - row_mean_transformer_c1) / denom_transformer_c1) * ln2_gain[ln2_offset_transformer_c1 + j_transformer_c1] + ln2_bias[ln2_offset_transformer_c1 + j_transformer_c1]
    return nothing
end

function jacc_kernel_transformer_loss_b_16!(__jacc_i, i_l_transformer_c1, n_d_transformer_c1, x, x_next, x_stack)
    idx_transformer_c1 = 1 + (__jacc_i - 1)
    __idx_x_stack_0 = ((i_l_transformer_c1 - 1) * (div(n_d_transformer_c1 - 1, 1) + 1) + (idx_transformer_c1 - 1)) + 1
    x_stack[__idx_x_stack_0] = x[idx_transformer_c1]
    x[idx_transformer_c1] = x_next[idx_transformer_c1]
    return nothing
end

function jacc_kernel_transformer_loss_b_17!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_transformer_loss_b_18!(__jacc_i, dk, h, loss, n, target, x)
    i_o = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += (x[i_o] - target[i_o]) ^ 2
    return nothing
end

function jacc_kernel_transformer_loss_b_19!(__jacc_i, dk, h, lossb, n, target, targetb, x, xb)
    i_o = n * h * dk + (__jacc_i - 1) * -1
    __cse_70 = (2 * (x[i_o] - target[i_o])) * lossb[1]
    xb[i_o] = xb[i_o] + __cse_70
    targetb[i_o] = targetb[i_o] + -__cse_70
    return nothing
end

function jacc_kernel_transformer_loss_b_20!(__jacc_i, i_l_transformer_c1, n_d_transformer_c1, x, x_nextb, x_stack, xb)
    idx_transformer_c1 = n_d_transformer_c1 + (__jacc_i - 1) * -1
    __idx_x_stack_0 = ((i_l_transformer_c1 - 1) * (div(n_d_transformer_c1 - 1, 1) + 1) + (idx_transformer_c1 - 1)) + 1
    x[idx_transformer_c1] = x_stack[__idx_x_stack_0]
    __oldb_2 = xb[idx_transformer_c1]
    xb[idx_transformer_c1] = 0.0
    x_nextb[idx_transformer_c1] = x_nextb[idx_transformer_c1] + __oldb_2
    return nothing
end

function jacc_kernel_transformer_loss_b_21!(__jacc_i, ff_outb, i_l_transformer_c1, n_d_transformer_c1, normed1b, resid2, resid2_stack, resid2b)
    idx_transformer_c1 = n_d_transformer_c1 + (__jacc_i - 1) * -1
    __idx_resid2_stack_0 = ((i_l_transformer_c1 - 1) * (div(n_d_transformer_c1 - 1, 1) + 1) + (idx_transformer_c1 - 1)) + 1
    resid2[idx_transformer_c1] = resid2_stack[__idx_resid2_stack_0]
    __oldb_2 = resid2b[idx_transformer_c1]
    resid2b[idx_transformer_c1] = 0.0
    normed1b[idx_transformer_c1] = normed1b[idx_transformer_c1] + __oldb_2
    ff_outb[idx_transformer_c1] = ff_outb[idx_transformer_c1] + __oldb_2
    return nothing
end

function jacc_kernel_transformer_loss_b_22!(__jacc_i, b2_offset_transformer_c1, b2b, d_transformer_c1, dff, dk, ff_hidden, ff_hiddenb, ff_outb, h, i_l_transformer_c1, n, n_d_transformer_c1, n_dff_transformer_c1, n_layers, s_transformer_c1_stack, w2, w2_offset_transformer_c1, w2b)
    idx_transformer_c1 = n_d_transformer_c1 + (__jacc_i - 1) * -1
    s_transformer_c1b = 0.0
    __icse_89 = max(0, div(n_layers - 1, 1) + 1)
    __icse_90 = div(n_d_transformer_c1 - 1, 1) + 1
    __icse_91 = __icse_89 * max(0, __icse_90)
    __icse_92 = max(0, div(h - 1, 1) + 1)
    __icse_93 = idx_transformer_c1 - 1
    __idx_s_transformer_c1_stack_0 = (((((((__icse_91 + __icse_91) + __icse_91) + __icse_89 * __icse_92 * max(0, div(n * n - 1, 1) + 1)) + __icse_89 * __icse_92 * max(0, div(n * dk - 1, 1) + 1)) + __icse_91) + __icse_89 * max(0, div(n - 1, 1) + 1)) + __icse_89 * max(0, div(n_dff_transformer_c1 - 1, 1) + 1)) + (((i_l_transformer_c1 - 1) * __icse_90 + __icse_93) + 1)
    s_transformer_c1 = s_transformer_c1_stack[__idx_s_transformer_c1_stack_0]
    i_transformer_c1 = div(__icse_93, d_transformer_c1) + 1
    j_transformer_c1 = mod(__icse_93, d_transformer_c1) + 1
    __oldb_0 = ff_outb[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
    ff_outb[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] = 0.0
    s_transformer_c1b = s_transformer_c1b + __oldb_0
    Atomix.@atomic b2b[b2_offset_transformer_c1 + j_transformer_c1] += __oldb_0
    for i_p_transformer_c1 = 1:dff
        __cse_94 = ff_hidden[(i_transformer_c1 - 1) * dff + i_p_transformer_c1]
        __cse_95 = w2[w2_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
        s_transformer_c1 = s_transformer_c1 + __cse_94 * __cse_95
        Atomix.@atomic ff_hiddenb[(i_transformer_c1 - 1) * dff + i_p_transformer_c1] += __cse_95 * s_transformer_c1b
        Atomix.@atomic w2b[w2_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] += __cse_94 * s_transformer_c1b
    end
    s_transformer_c1b = 0.0
    return nothing
end

function jacc_kernel_transformer_loss_b_23!(__jacc_i, b1, b1_offset_transformer_c1, b1b, d_transformer_c1, dff, dk, ff_hidden, ff_hidden_stack, ff_hiddenb, h, i_l_transformer_c1, n, n_d_transformer_c1, n_dff_transformer_c1, n_layers, normed1, normed1b, s_transformer_c1_stack, w1, w1_offset_transformer_c1, w1b)
    idx_transformer_c1 = n_dff_transformer_c1 + (__jacc_i - 1) * -1
    s_transformer_c1b = 0.0
    __icse_96 = max(0, div(n_layers - 1, 1) + 1)
    __icse_97 = __icse_96 * max(0, div(n_d_transformer_c1 - 1, 1) + 1)
    __icse_98 = max(0, div(h - 1, 1) + 1)
    __icse_99 = idx_transformer_c1 - 1
    __icse_100 = ((i_l_transformer_c1 - 1) * (div(n_dff_transformer_c1 - 1, 1) + 1) + __icse_99) + 1
    __idx_s_transformer_c1_stack_0 = ((((((__icse_97 + __icse_97) + __icse_97) + __icse_96 * __icse_98 * max(0, div(n * n - 1, 1) + 1)) + __icse_96 * __icse_98 * max(0, div(n * dk - 1, 1) + 1)) + __icse_97) + __icse_96 * max(0, div(n - 1, 1) + 1)) + __icse_100
    s_transformer_c1 = s_transformer_c1_stack[__idx_s_transformer_c1_stack_0]
    i_transformer_c1 = div(__icse_99, dff) + 1
    j_transformer_c1 = mod(__icse_99, dff) + 1
    __idx_ff_hidden_stack_0 = __icse_100
    ff_hidden[(i_transformer_c1 - 1) * dff + j_transformer_c1] = ff_hidden_stack[__idx_ff_hidden_stack_0]
    __oldb_2 = ff_hiddenb[(i_transformer_c1 - 1) * dff + j_transformer_c1]
    ff_hiddenb[(i_transformer_c1 - 1) * dff + j_transformer_c1] = 0.0
    __cse_101 = (0.5 * (1.0 + sign(s_transformer_c1 + b1[b1_offset_transformer_c1 + j_transformer_c1]))) * __oldb_2
    s_transformer_c1b = s_transformer_c1b + __cse_101
    Atomix.@atomic b1b[b1_offset_transformer_c1 + j_transformer_c1] += __cse_101
    for i_p_transformer_c1 = 1:d_transformer_c1
        __cse_102 = normed1[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1]
        __cse_103 = w1[w1_offset_transformer_c1 + (i_p_transformer_c1 - 1) * dff + j_transformer_c1]
        s_transformer_c1 = s_transformer_c1 + __cse_102 * __cse_103
        Atomix.@atomic normed1b[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1] += __cse_103 * s_transformer_c1b
        Atomix.@atomic w1b[w1_offset_transformer_c1 + (i_p_transformer_c1 - 1) * dff + j_transformer_c1] += __cse_102 * s_transformer_c1b
    end
    s_transformer_c1b = 0.0
    return nothing
end

function jacc_kernel_transformer_loss_b_24!(__jacc_i, attn_outb, i_l_transformer_c1, n_d_transformer_c1, resid1, resid1_stack, resid1b, xb)
    idx_transformer_c1 = n_d_transformer_c1 + (__jacc_i - 1) * -1
    __idx_resid1_stack_0 = ((i_l_transformer_c1 - 1) * (div(n_d_transformer_c1 - 1, 1) + 1) + (idx_transformer_c1 - 1)) + 1
    resid1[idx_transformer_c1] = resid1_stack[__idx_resid1_stack_0]
    __oldb_2 = resid1b[idx_transformer_c1]
    resid1b[idx_transformer_c1] = 0.0
    xb[idx_transformer_c1] = xb[idx_transformer_c1] + __oldb_2
    attn_outb[idx_transformer_c1] = attn_outb[idx_transformer_c1] + __oldb_2
    return nothing
end

function jacc_kernel_transformer_loss_b_25!(__jacc_i, attn_outb, b_offset_transformer_c1, bob, ctx, ctxb, d_transformer_c1, dk, h, i_l_transformer_c1, n, n_d_transformer_c1, n_layers, s_transformer_c1_stack, w_offset_transformer_c1, wo, wob)
    idx_transformer_c1 = n_d_transformer_c1 + (__jacc_i - 1) * -1
    s_transformer_c1b = 0.0
    __icse_116 = max(0, div(n_layers - 1, 1) + 1)
    __icse_117 = div(n_d_transformer_c1 - 1, 1) + 1
    __icse_118 = __icse_116 * max(0, __icse_117)
    __icse_119 = max(0, div(h - 1, 1) + 1)
    __icse_120 = idx_transformer_c1 - 1
    __idx_s_transformer_c1_stack_0 = ((((__icse_118 + __icse_118) + __icse_118) + __icse_116 * __icse_119 * max(0, div(n * n - 1, 1) + 1)) + __icse_116 * __icse_119 * max(0, div(n * dk - 1, 1) + 1)) + (((i_l_transformer_c1 - 1) * __icse_117 + __icse_120) + 1)
    s_transformer_c1 = s_transformer_c1_stack[__idx_s_transformer_c1_stack_0]
    i_transformer_c1 = div(__icse_120, d_transformer_c1) + 1
    j_transformer_c1 = mod(__icse_120, d_transformer_c1) + 1
    __oldb_0 = attn_outb[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
    attn_outb[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] = 0.0
    s_transformer_c1b = s_transformer_c1b + __oldb_0
    Atomix.@atomic bob[b_offset_transformer_c1 + j_transformer_c1] += __oldb_0
    for i_p_transformer_c1 = 1:d_transformer_c1
        __cse_121 = ctx[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1]
        __cse_122 = wo[w_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
        s_transformer_c1 = s_transformer_c1 + __cse_121 * __cse_122
        Atomix.@atomic ctxb[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1] += __cse_122 * s_transformer_c1b
        Atomix.@atomic wob[w_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] += __cse_121 * s_transformer_c1b
    end
    s_transformer_c1b = 0.0
    return nothing
end

function jacc_kernel_transformer_loss_b_26!(__jacc_i, ctx, ctx_stack, ctxb, d_transformer_c1, dk, h, head_offset_transformer_c1, hh_transformer_c1, i_l_transformer_c1, n, n_d_transformer_c1, n_layers, probs, probsb, s_transformer_c1_stack, score_off_transformer_c1, v, vb)
    idx3_transformer_c1 = n * dk + (__jacc_i - 1) * -1
    s_transformer_c1b = 0.0
    __icse_124 = max(0, div(n_layers - 1, 1) + 1)
    __icse_125 = __icse_124 * max(0, div(n_d_transformer_c1 - 1, 1) + 1)
    __icse_126 = div(h - 1, 1) + 1
    __icse_127 = div(n * dk - 1, 1) + 1
    __icse_128 = idx3_transformer_c1 - 1
    __icse_129 = ((i_l_transformer_c1 - 1) * (__icse_126 * __icse_127) + (hh_transformer_c1 - 1) * __icse_127 + __icse_128) + 1
    __idx_s_transformer_c1_stack_0 = (((__icse_125 + __icse_125) + __icse_125) + __icse_124 * max(0, __icse_126) * max(0, div(n * n - 1, 1) + 1)) + __icse_129
    s_transformer_c1 = s_transformer_c1_stack[__idx_s_transformer_c1_stack_0]
    i_transformer_c1 = div(__icse_128, dk) + 1
    p_transformer_c1 = mod(__icse_128, dk) + 1
    __idx_ctx_stack_0 = __icse_129
    ctx[(i_transformer_c1 - 1) * d_transformer_c1 + head_offset_transformer_c1 + p_transformer_c1] = ctx_stack[__idx_ctx_stack_0]
    __oldb_2 = ctxb[(i_transformer_c1 - 1) * d_transformer_c1 + head_offset_transformer_c1 + p_transformer_c1]
    ctxb[(i_transformer_c1 - 1) * d_transformer_c1 + head_offset_transformer_c1 + p_transformer_c1] = 0.0
    s_transformer_c1b = s_transformer_c1b + __oldb_2
    for i_j_transformer_c1 = 1:n
        __cse_130 = probs[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + i_j_transformer_c1]
        __cse_131 = v[(i_j_transformer_c1 - 1) * d_transformer_c1 + head_offset_transformer_c1 + p_transformer_c1]
        s_transformer_c1 = s_transformer_c1 + __cse_130 * __cse_131
        Atomix.@atomic probsb[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + i_j_transformer_c1] += __cse_131 * s_transformer_c1b
        Atomix.@atomic vb[(i_j_transformer_c1 - 1) * d_transformer_c1 + head_offset_transformer_c1 + p_transformer_c1] += __cse_130 * s_transformer_c1b
    end
    s_transformer_c1b = 0.0
    return nothing
end

function jacc_kernel_transformer_loss_b_27!(__jacc_i, d_transformer_c1, dk, h, head_offset_transformer_c1, hh_transformer_c1, i_l_transformer_c1, inv_sqrt_dk_transformer_c1, k, kb, n, n_d_transformer_c1, n_layers, q, qb, s_transformer_c1_stack, score_off_transformer_c1, scores, scores_stack, scoresb)
    idx2_transformer_c1 = n * n + (__jacc_i - 1) * -1
    s_transformer_c1b = 0.0
    __icse_145 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d_transformer_c1 - 1, 1) + 1)
    __icse_146 = div(n * n - 1, 1) + 1
    __icse_147 = idx2_transformer_c1 - 1
    __icse_148 = ((i_l_transformer_c1 - 1) * ((div(h - 1, 1) + 1) * __icse_146) + (hh_transformer_c1 - 1) * __icse_146 + __icse_147) + 1
    __idx_s_transformer_c1_stack_0 = ((__icse_145 + __icse_145) + __icse_145) + __icse_148
    s_transformer_c1 = s_transformer_c1_stack[__idx_s_transformer_c1_stack_0]
    i_transformer_c1 = div(__icse_147, n) + 1
    j_transformer_c1 = mod(__icse_147, n) + 1
    __idx_scores_stack_0 = __icse_148
    scores[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + j_transformer_c1] = scores_stack[__idx_scores_stack_0]
    __oldb_2 = scoresb[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + j_transformer_c1]
    scoresb[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + j_transformer_c1] = 0.0
    s_transformer_c1b = s_transformer_c1b + inv_sqrt_dk_transformer_c1 * __oldb_2
    for i_p_transformer_c1 = 1:dk
        __cse_149 = q[(i_transformer_c1 - 1) * d_transformer_c1 + head_offset_transformer_c1 + i_p_transformer_c1]
        __cse_150 = k[(j_transformer_c1 - 1) * d_transformer_c1 + head_offset_transformer_c1 + i_p_transformer_c1]
        s_transformer_c1 = s_transformer_c1 + __cse_149 * __cse_150
        Atomix.@atomic qb[(i_transformer_c1 - 1) * d_transformer_c1 + head_offset_transformer_c1 + i_p_transformer_c1] += __cse_150 * s_transformer_c1b
        Atomix.@atomic kb[(j_transformer_c1 - 1) * d_transformer_c1 + head_offset_transformer_c1 + i_p_transformer_c1] += __cse_149 * s_transformer_c1b
    end
    s_transformer_c1b = 0.0
    return nothing
end

function jacc_kernel_transformer_loss_b_28!(__jacc_i, b_offset_transformer_c1, bvb, d_transformer_c1, i_l_transformer_c1, n_d_transformer_c1, n_layers, s_transformer_c1_stack, v, v_stack, vb, w_offset_transformer_c1, wv, wvb, x, xb)
    idx_transformer_c1 = n_d_transformer_c1 + (__jacc_i - 1) * -1
    s_transformer_c1b = 0.0
    __icse_151 = div(n_d_transformer_c1 - 1, 1) + 1
    __icse_152 = max(0, div(n_layers - 1, 1) + 1) * max(0, __icse_151)
    __icse_153 = idx_transformer_c1 - 1
    __icse_154 = ((i_l_transformer_c1 - 1) * __icse_151 + __icse_153) + 1
    __idx_s_transformer_c1_stack_0 = (__icse_152 + __icse_152) + __icse_154
    s_transformer_c1 = s_transformer_c1_stack[__idx_s_transformer_c1_stack_0]
    i_transformer_c1 = div(__icse_153, d_transformer_c1) + 1
    j_transformer_c1 = mod(__icse_153, d_transformer_c1) + 1
    __idx_v_stack_0 = __icse_154
    v[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] = v_stack[__idx_v_stack_0]
    __oldb_2 = vb[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
    vb[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] = 0.0
    s_transformer_c1b = s_transformer_c1b + __oldb_2
    Atomix.@atomic bvb[b_offset_transformer_c1 + j_transformer_c1] += __oldb_2
    for i_p_transformer_c1 = 1:d_transformer_c1
        __cse_155 = x[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1]
        __cse_156 = wv[w_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
        s_transformer_c1 = s_transformer_c1 + __cse_155 * __cse_156
        Atomix.@atomic xb[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1] += __cse_156 * s_transformer_c1b
        Atomix.@atomic wvb[w_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] += __cse_155 * s_transformer_c1b
    end
    s_transformer_c1b = 0.0
    return nothing
end

function jacc_kernel_transformer_loss_b_29!(__jacc_i, b_offset_transformer_c1, bkb, d_transformer_c1, i_l_transformer_c1, k, k_stack, kb, n_d_transformer_c1, n_layers, s_transformer_c1_stack, w_offset_transformer_c1, wk, wkb, x, xb)
    idx_transformer_c1 = n_d_transformer_c1 + (__jacc_i - 1) * -1
    s_transformer_c1b = 0.0
    __icse_157 = div(n_d_transformer_c1 - 1, 1) + 1
    __icse_158 = idx_transformer_c1 - 1
    __icse_159 = ((i_l_transformer_c1 - 1) * __icse_157 + __icse_158) + 1
    __idx_s_transformer_c1_stack_0 = max(0, div(n_layers - 1, 1) + 1) * max(0, __icse_157) + __icse_159
    s_transformer_c1 = s_transformer_c1_stack[__idx_s_transformer_c1_stack_0]
    i_transformer_c1 = div(__icse_158, d_transformer_c1) + 1
    j_transformer_c1 = mod(__icse_158, d_transformer_c1) + 1
    __idx_k_stack_0 = __icse_159
    k[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] = k_stack[__idx_k_stack_0]
    __oldb_2 = kb[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
    kb[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] = 0.0
    s_transformer_c1b = s_transformer_c1b + __oldb_2
    Atomix.@atomic bkb[b_offset_transformer_c1 + j_transformer_c1] += __oldb_2
    for i_p_transformer_c1 = 1:d_transformer_c1
        __cse_160 = x[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1]
        __cse_161 = wk[w_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
        s_transformer_c1 = s_transformer_c1 + __cse_160 * __cse_161
        Atomix.@atomic xb[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1] += __cse_161 * s_transformer_c1b
        Atomix.@atomic wkb[w_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] += __cse_160 * s_transformer_c1b
    end
    s_transformer_c1b = 0.0
    return nothing
end

function jacc_kernel_transformer_loss_b_30!(__jacc_i, b_offset_transformer_c1, bqb, d_transformer_c1, i_l_transformer_c1, n_d_transformer_c1, q, q_stack, qb, s_transformer_c1_stack, w_offset_transformer_c1, wq, wqb, x, xb)
    idx_transformer_c1 = n_d_transformer_c1 + (__jacc_i - 1) * -1
    s_transformer_c1b = 0.0
    __icse_162 = idx_transformer_c1 - 1
    __icse_163 = ((i_l_transformer_c1 - 1) * (div(n_d_transformer_c1 - 1, 1) + 1) + __icse_162) + 1
    __idx_s_transformer_c1_stack_0 = __icse_163
    s_transformer_c1 = s_transformer_c1_stack[__idx_s_transformer_c1_stack_0]
    i_transformer_c1 = div(__icse_162, d_transformer_c1) + 1
    j_transformer_c1 = mod(__icse_162, d_transformer_c1) + 1
    __idx_q_stack_0 = __icse_163
    q[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] = q_stack[__idx_q_stack_0]
    __oldb_2 = qb[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
    qb[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] = 0.0
    s_transformer_c1b = s_transformer_c1b + __oldb_2
    Atomix.@atomic bqb[b_offset_transformer_c1 + j_transformer_c1] += __oldb_2
    for i_p_transformer_c1 = 1:d_transformer_c1
        __cse_164 = x[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1]
        __cse_165 = wq[w_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
        s_transformer_c1 = s_transformer_c1 + __cse_164 * __cse_165
        Atomix.@atomic xb[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1] += __cse_165 * s_transformer_c1b
        Atomix.@atomic wqb[w_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] += __cse_164 * s_transformer_c1b
    end
    s_transformer_c1b = 0.0
    return nothing
end

function jacc_kernel_transformer_loss_b_31!(__jacc_i, dk, h, n, x_inb, xb)
    i_c = n * h * dk + (__jacc_i - 1) * -1
    __oldb_0 = xb[i_c]
    xb[i_c] = 0.0
    x_inb[i_c] = x_inb[i_c] + __oldb_0
    return nothing
end

function jacc_kernel_transformer_loss_1!(__jacc_i, dk, h, n, x, x_in)
    i_c = 1 + (__jacc_i - 1)
    x[i_c] = x_in[i_c]
    return nothing
end

function jacc_kernel_transformer_loss_2!(__jacc_i, b_offset_transformer_c1, bq, d_transformer_c1, n_d_transformer_c1, q, w_offset_transformer_c1, wq, x)
    idx_transformer_c1 = 1 + (__jacc_i - 1)
    i_transformer_c1 = div(idx_transformer_c1 - 1, d_transformer_c1) + 1
    j_transformer_c1 = mod(idx_transformer_c1 - 1, d_transformer_c1) + 1
    s_transformer_c1 = 0.0
    for i_p_transformer_c1 = 1:d_transformer_c1
        s_transformer_c1 = s_transformer_c1 + x[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1] * wq[w_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
    end
    q[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] = s_transformer_c1 + bq[b_offset_transformer_c1 + j_transformer_c1]
    return nothing
end

function jacc_kernel_transformer_loss_3!(__jacc_i, b_offset_transformer_c1, bk, d_transformer_c1, k, n_d_transformer_c1, w_offset_transformer_c1, wk, x)
    idx_transformer_c1 = 1 + (__jacc_i - 1)
    i_transformer_c1 = div(idx_transformer_c1 - 1, d_transformer_c1) + 1
    j_transformer_c1 = mod(idx_transformer_c1 - 1, d_transformer_c1) + 1
    s_transformer_c1 = 0.0
    for i_p_transformer_c1 = 1:d_transformer_c1
        s_transformer_c1 = s_transformer_c1 + x[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1] * wk[w_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
    end
    k[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] = s_transformer_c1 + bk[b_offset_transformer_c1 + j_transformer_c1]
    return nothing
end

function jacc_kernel_transformer_loss_4!(__jacc_i, b_offset_transformer_c1, bv, d_transformer_c1, n_d_transformer_c1, v, w_offset_transformer_c1, wv, x)
    idx_transformer_c1 = 1 + (__jacc_i - 1)
    i_transformer_c1 = div(idx_transformer_c1 - 1, d_transformer_c1) + 1
    j_transformer_c1 = mod(idx_transformer_c1 - 1, d_transformer_c1) + 1
    s_transformer_c1 = 0.0
    for i_p_transformer_c1 = 1:d_transformer_c1
        s_transformer_c1 = s_transformer_c1 + x[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1] * wv[w_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
    end
    v[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] = s_transformer_c1 + bv[b_offset_transformer_c1 + j_transformer_c1]
    return nothing
end

function jacc_kernel_transformer_loss_5!(__jacc_i, d_transformer_c1, dk, head_offset_transformer_c1, inv_sqrt_dk_transformer_c1, k, n, q, score_off_transformer_c1, scores)
    idx2_transformer_c1 = 1 + (__jacc_i - 1)
    i_transformer_c1 = div(idx2_transformer_c1 - 1, n) + 1
    j_transformer_c1 = mod(idx2_transformer_c1 - 1, n) + 1
    s_transformer_c1 = 0.0
    for i_p_transformer_c1 = 1:dk
        s_transformer_c1 = s_transformer_c1 + q[(i_transformer_c1 - 1) * d_transformer_c1 + head_offset_transformer_c1 + i_p_transformer_c1] * k[(j_transformer_c1 - 1) * d_transformer_c1 + head_offset_transformer_c1 + i_p_transformer_c1]
    end
    scores[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + j_transformer_c1] = s_transformer_c1 * inv_sqrt_dk_transformer_c1
    return nothing
end

function jacc_kernel_transformer_loss_6!(__jacc_i, i_transformer_c1, n, probs, row_max_transformer_c1, score_off_transformer_c1, scores)
    j_transformer_c1 = 1 + (__jacc_i - 1)
    kk_transformer_c1 = score_off_transformer_c1 + (i_transformer_c1 - 1) * n + j_transformer_c1
    probs[kk_transformer_c1] = exp(scores[kk_transformer_c1] - row_max_transformer_c1)
    return nothing
end

function jacc_kernel_transformer_loss_7!(__jacc_i, i_transformer_c1, n, probs, row_sum_transformer_c1, score_off_transformer_c1)
    j_transformer_c1 = 1 + (__jacc_i - 1)
    kk_transformer_c1 = score_off_transformer_c1 + (i_transformer_c1 - 1) * n + j_transformer_c1
    probs[kk_transformer_c1] = probs[kk_transformer_c1] / row_sum_transformer_c1
    return nothing
end

function jacc_kernel_transformer_loss_8!(__jacc_i, ctx, d_transformer_c1, dk, head_offset_transformer_c1, n, probs, score_off_transformer_c1, v)
    idx3_transformer_c1 = 1 + (__jacc_i - 1)
    i_transformer_c1 = div(idx3_transformer_c1 - 1, dk) + 1
    p_transformer_c1 = mod(idx3_transformer_c1 - 1, dk) + 1
    s_transformer_c1 = 0.0
    for i_j_transformer_c1 = 1:n
        s_transformer_c1 = s_transformer_c1 + probs[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + i_j_transformer_c1] * v[(i_j_transformer_c1 - 1) * d_transformer_c1 + head_offset_transformer_c1 + p_transformer_c1]
    end
    ctx[(i_transformer_c1 - 1) * d_transformer_c1 + head_offset_transformer_c1 + p_transformer_c1] = s_transformer_c1
    return nothing
end

function jacc_kernel_transformer_loss_9!(__jacc_i, attn_out, b_offset_transformer_c1, bo, ctx, d_transformer_c1, n_d_transformer_c1, w_offset_transformer_c1, wo)
    idx_transformer_c1 = 1 + (__jacc_i - 1)
    i_transformer_c1 = div(idx_transformer_c1 - 1, d_transformer_c1) + 1
    j_transformer_c1 = mod(idx_transformer_c1 - 1, d_transformer_c1) + 1
    s_transformer_c1 = 0.0
    for i_p_transformer_c1 = 1:d_transformer_c1
        s_transformer_c1 = s_transformer_c1 + ctx[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1] * wo[w_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
    end
    attn_out[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] = s_transformer_c1 + bo[b_offset_transformer_c1 + j_transformer_c1]
    return nothing
end

function jacc_kernel_transformer_loss_10!(__jacc_i, attn_out, n_d_transformer_c1, resid1, x)
    idx_transformer_c1 = 1 + (__jacc_i - 1)
    resid1[idx_transformer_c1] = x[idx_transformer_c1] + attn_out[idx_transformer_c1]
    return nothing
end

function jacc_kernel_transformer_loss_11!(__jacc_i, d_transformer_c1, eps, ln1_bias, ln1_gain, ln_offset_transformer_c1, n, normed1, resid1)
    i_transformer_c1 = 1 + (__jacc_i - 1)
    s_transformer_c1 = 0.0
    for i_j_transformer_c1 = 1:d_transformer_c1
        s_transformer_c1 = s_transformer_c1 + resid1[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1]
    end
    row_mean_transformer_c1 = s_transformer_c1 / d_transformer_c1
    s2_transformer_c1 = 0.0
    for i_j_transformer_c1 = 1:d_transformer_c1
        diff_transformer_c1 = resid1[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1] - row_mean_transformer_c1
        s2_transformer_c1 = s2_transformer_c1 + diff_transformer_c1 * diff_transformer_c1
    end
    row_var_transformer_c1 = s2_transformer_c1 / d_transformer_c1
    denom_transformer_c1 = sqrt(row_var_transformer_c1 + eps)
    for j_transformer_c1 = 1:d_transformer_c1
        kk_transformer_c1 = (i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1
        normed1[kk_transformer_c1] = ((resid1[kk_transformer_c1] - row_mean_transformer_c1) / denom_transformer_c1) * ln1_gain[ln_offset_transformer_c1 + j_transformer_c1] + ln1_bias[ln_offset_transformer_c1 + j_transformer_c1]
    end
    return nothing
end

function jacc_kernel_transformer_loss_12!(__jacc_i, b1, b1_offset_transformer_c1, d_transformer_c1, dff, ff_hidden, n_dff_transformer_c1, normed1, w1, w1_offset_transformer_c1)
    idx_transformer_c1 = 1 + (__jacc_i - 1)
    i_transformer_c1 = div(idx_transformer_c1 - 1, dff) + 1
    j_transformer_c1 = mod(idx_transformer_c1 - 1, dff) + 1
    s_transformer_c1 = 0.0
    for i_p_transformer_c1 = 1:d_transformer_c1
        s_transformer_c1 = s_transformer_c1 + normed1[(i_transformer_c1 - 1) * d_transformer_c1 + i_p_transformer_c1] * w1[w1_offset_transformer_c1 + (i_p_transformer_c1 - 1) * dff + j_transformer_c1]
    end
    ff_hidden[(i_transformer_c1 - 1) * dff + j_transformer_c1] = max(s_transformer_c1 + b1[b1_offset_transformer_c1 + j_transformer_c1], 0.0)
    return nothing
end

function jacc_kernel_transformer_loss_13!(__jacc_i, b2, b2_offset_transformer_c1, d_transformer_c1, dff, ff_hidden, ff_out, n_d_transformer_c1, w2, w2_offset_transformer_c1)
    idx_transformer_c1 = 1 + (__jacc_i - 1)
    i_transformer_c1 = div(idx_transformer_c1 - 1, d_transformer_c1) + 1
    j_transformer_c1 = mod(idx_transformer_c1 - 1, d_transformer_c1) + 1
    s_transformer_c1 = 0.0
    for i_p_transformer_c1 = 1:dff
        s_transformer_c1 = s_transformer_c1 + ff_hidden[(i_transformer_c1 - 1) * dff + i_p_transformer_c1] * w2[w2_offset_transformer_c1 + (i_p_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1]
    end
    ff_out[(i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1] = s_transformer_c1 + b2[b2_offset_transformer_c1 + j_transformer_c1]
    return nothing
end

function jacc_kernel_transformer_loss_14!(__jacc_i, ff_out, n_d_transformer_c1, normed1, resid2)
    idx_transformer_c1 = 1 + (__jacc_i - 1)
    resid2[idx_transformer_c1] = normed1[idx_transformer_c1] + ff_out[idx_transformer_c1]
    return nothing
end

function jacc_kernel_transformer_loss_15!(__jacc_i, d_transformer_c1, eps, ln2_bias, ln2_gain, ln2_offset_transformer_c1, n, resid2, x_next)
    i_transformer_c1 = 1 + (__jacc_i - 1)
    s_transformer_c1 = 0.0
    for i_j_transformer_c1 = 1:d_transformer_c1
        s_transformer_c1 = s_transformer_c1 + resid2[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1]
    end
    row_mean_transformer_c1 = s_transformer_c1 / d_transformer_c1
    s2_transformer_c1 = 0.0
    for i_j_transformer_c1 = 1:d_transformer_c1
        diff_transformer_c1 = resid2[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1] - row_mean_transformer_c1
        s2_transformer_c1 = s2_transformer_c1 + diff_transformer_c1 * diff_transformer_c1
    end
    row_var_transformer_c1 = s2_transformer_c1 / d_transformer_c1
    denom_transformer_c1 = sqrt(row_var_transformer_c1 + eps)
    for j_transformer_c1 = 1:d_transformer_c1
        kk_transformer_c1 = (i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1
        x_next[kk_transformer_c1] = ((resid2[kk_transformer_c1] - row_mean_transformer_c1) / denom_transformer_c1) * ln2_gain[ln2_offset_transformer_c1 + j_transformer_c1] + ln2_bias[ln2_offset_transformer_c1 + j_transformer_c1]
    end
    return nothing
end

function jacc_kernel_transformer_loss_16!(__jacc_i, n_d_transformer_c1, x, x_next)
    idx_transformer_c1 = 1 + (__jacc_i - 1)
    x[idx_transformer_c1] = x_next[idx_transformer_c1]
    return nothing
end

function jacc_kernel_transformer_loss_17!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_transformer_loss_18!(__jacc_i, dk, h, loss, n, target, x)
    i_o = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += (x[i_o] - target[i_o]) ^ 2
    return nothing
end

function initstacks_transformer_loss_b_jacc(dff, dk, h, n, n_layers)
    d_transformer_c1 = h * dk
    n_d_transformer_c1 = n * d_transformer_c1
    n_dff_transformer_c1 = n * dff
    q_stack = JACC.zeros(Float64, max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d_transformer_c1 - 1, 1) + 1))
    k_stack = JACC.zeros(Float64, max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d_transformer_c1 - 1, 1) + 1))
    v_stack = JACC.zeros(Float64, max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d_transformer_c1 - 1, 1) + 1))
    scores_stack = JACC.zeros(Float64, max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * n - 1, 1) + 1))
    row_max_transformer_c1_stack = JACC.zeros(Float64, max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) * max(0, div(n - 2, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n - 1, 1) + 1))
    probs_stack = JACC.zeros(Float64, max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) * max(0, div(n - 1, 1) + 1))
    ctx_stack = JACC.zeros(Float64, max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * dk - 1, 1) + 1))
    resid1_stack = JACC.zeros(Float64, max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d_transformer_c1 - 1, 1) + 1))
    row_mean_transformer_c1_stack = JACC.zeros(Float64, ((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1)) + 1)
    row_var_transformer_c1_stack = JACC.zeros(Float64, ((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1)) + 1)
    denom_transformer_c1_stack = JACC.zeros(Float64, ((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1)) + 1)
    normed1_stack = JACC.zeros(Float64, max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) * max(0, div(d_transformer_c1 - 1, 1) + 1))
    ff_hidden_stack = JACC.zeros(Float64, max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_dff_transformer_c1 - 1, 1) + 1))
    resid2_stack = JACC.zeros(Float64, max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d_transformer_c1 - 1, 1) + 1))
    x_stack = JACC.zeros(Float64, max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d_transformer_c1 - 1, 1) + 1))
    row_sum_transformer_c1_stack = JACC.zeros(Float64, max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n - 1, 1) + 1))
    s2_transformer_c1_stack = JACC.zeros(Float64, max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1))
    s_transformer_c1_stack = JACC.zeros(Float64, ((((((((max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d_transformer_c1 - 1, 1) + 1) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d_transformer_c1 - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d_transformer_c1 - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(h - 1, 1) + 1) * max(0, div(n * dk - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d_transformer_c1 - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_dff_transformer_c1 - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n_d_transformer_c1 - 1, 1) + 1)) + max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1))
    return (q_stack, k_stack, v_stack, scores_stack, row_max_transformer_c1_stack, probs_stack, ctx_stack, resid1_stack, row_mean_transformer_c1_stack, row_var_transformer_c1_stack, denom_transformer_c1_stack, normed1_stack, ff_hidden_stack, resid2_stack, x_stack, row_sum_transformer_c1_stack, s2_transformer_c1_stack, s_transformer_c1_stack)
end

function transformer_loss_b_jacc(x, xb, wq, wqb, bq, bqb, wk, wkb, bk, bkb, wv, wvb, bv, bvb, wo, wob, bo, bob, ln1_gain, ln1_gainb, ln1_bias, ln1_biasb, w1, w1b, b1, b1b, w2, w2b, b2, b2b, ln2_gain, ln2_gainb, ln2_bias, ln2_biasb, q, qb, k, kb, v, vb, scores, scoresb, probs, probsb, ctx, ctxb, attn_out, attn_outb, resid1, resid1b, normed1, normed1b, ff_hidden, ff_hiddenb, ff_out, ff_outb, resid2, resid2b, x_next, x_nextb, n, dk, h, dff, n_layers, eps, epsb, x_in, x_inb, target, targetb, loss, lossb, q_stack, k_stack, v_stack, scores_stack, row_max_transformer_c1_stack, probs_stack, ctx_stack, resid1_stack, row_mean_transformer_c1_stack, row_var_transformer_c1_stack, denom_transformer_c1_stack, normed1_stack, ff_hidden_stack, resid2_stack, x_stack, row_sum_transformer_c1_stack, s2_transformer_c1_stack, s_transformer_c1_stack)
    denom_transformer_c1 = 0.0
    diff_transformer_c1 = 0.0
    row_max_transformer_c1 = 0.0
    row_mean_transformer_c1 = 0.0
    row_sum_transformer_c1 = 0.0
    row_var_transformer_c1 = 0.0
    s2_transformer_c1 = 0.0
    s_transformer_c1 = 0.0
    denom_transformer_c1b = 0.0
    diff_transformer_c1b = 0.0
    row_max_transformer_c1b = 0.0
    row_mean_transformer_c1b = 0.0
    row_sum_transformer_c1b = 0.0
    row_var_transformer_c1b = 0.0
    s2_transformer_c1b = 0.0
    s_transformer_c1b = 0.0
    if div(n * h * dk - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n * h * dk - 1, 1) + 1 jacc_kernel_transformer_loss_b_1!(dk, h, n, x, x_in)
    end
    d_transformer_c1 = h * dk
    inv_sqrt_dk_transformer_c1 = 1.0 / sqrt(dk)
    n_d_transformer_c1 = n * d_transformer_c1
    n_dff_transformer_c1 = n * dff
    for i_l_transformer_c1 = 1:n_layers
        __icse_0 = i_l_transformer_c1 - 1
        w_offset_transformer_c1 = __icse_0 * d_transformer_c1 * d_transformer_c1
        __icse_1 = __icse_0 * d_transformer_c1
        b_offset_transformer_c1 = __icse_1
        ln_offset_transformer_c1 = __icse_1
        w1_offset_transformer_c1 = __icse_0 * d_transformer_c1 * dff
        b1_offset_transformer_c1 = __icse_0 * dff
        w2_offset_transformer_c1 = __icse_0 * dff * d_transformer_c1
        b2_offset_transformer_c1 = __icse_1
        ln2_offset_transformer_c1 = __icse_1
        if div(n_d_transformer_c1 - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_b_2!(b_offset_transformer_c1, bq, d_transformer_c1, i_l_transformer_c1, n_d_transformer_c1, q, q_stack, s_transformer_c1_stack, w_offset_transformer_c1, wq, x)
        end
        if div(n_d_transformer_c1 - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_b_3!(b_offset_transformer_c1, bk, d_transformer_c1, i_l_transformer_c1, k, k_stack, n_d_transformer_c1, n_layers, s_transformer_c1_stack, w_offset_transformer_c1, wk, x)
        end
        if div(n_d_transformer_c1 - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_b_4!(b_offset_transformer_c1, bv, d_transformer_c1, i_l_transformer_c1, n_d_transformer_c1, n_layers, s_transformer_c1_stack, v, v_stack, w_offset_transformer_c1, wv, x)
        end
        for hh_transformer_c1 = 1:h
            __icse_11 = hh_transformer_c1 - 1
            head_offset_transformer_c1 = __icse_11 * dk
            score_off_transformer_c1 = __icse_11 * n * n
            if div(n * n - 1, 1) + 1 > 0
                JACC.@parallel_for range = div(n * n - 1, 1) + 1 jacc_kernel_transformer_loss_b_5!(d_transformer_c1, dk, h, head_offset_transformer_c1, hh_transformer_c1, i_l_transformer_c1, inv_sqrt_dk_transformer_c1, k, n, n_d_transformer_c1, n_layers, q, s_transformer_c1_stack, score_off_transformer_c1, scores, scores_stack)
            end
            for i_transformer_c1 = 1:n
                CUDA.@allowscalar begin
                        row_max_transformer_c1 = scores[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + 1]
                    end
                for i_j_transformer_c1 = 2:n
                    CUDA.@allowscalar begin
                            __icse_16 = div(n - 1, 1) + 1
                            __icse_17 = div(n - 2, 1) + 1
                            __idx_row_max_transformer_c1_stack_0 = ((i_l_transformer_c1 - 1) * ((div(h - 1, 1) + 1) * __icse_16 * __icse_17) + (hh_transformer_c1 - 1) * (__icse_16 * __icse_17) + (i_transformer_c1 - 1) * __icse_17 + (i_j_transformer_c1 - 2)) + 1
                            row_max_transformer_c1_stack[__idx_row_max_transformer_c1_stack_0] = row_max_transformer_c1
                            row_max_transformer_c1 = max(row_max_transformer_c1, scores[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + i_j_transformer_c1])
                        end
                end
                if div(n - 1, 1) + 1 > 0
                    JACC.@parallel_for range = div(n - 1, 1) + 1 jacc_kernel_transformer_loss_b_6!(h, hh_transformer_c1, i_l_transformer_c1, i_transformer_c1, n, probs, probs_stack, row_max_transformer_c1, score_off_transformer_c1, scores)
                end
                row_sum_transformer_c1 = 0.0
                for i_j_transformer_c1 = 1:n
                    CUDA.@allowscalar begin
                            row_sum_transformer_c1 = row_sum_transformer_c1 + probs[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + i_j_transformer_c1]
                        end
                end
                if div(n - 1, 1) + 1 > 0
                    JACC.@parallel_for range = div(n - 1, 1) + 1 jacc_kernel_transformer_loss_b_7!(h, hh_transformer_c1, i_l_transformer_c1, i_transformer_c1, n, n_layers, probs, probs_stack, row_sum_transformer_c1, score_off_transformer_c1)
                end
                CUDA.@allowscalar begin
                        __icse_25 = div(h - 1, 1) + 1
                        __icse_26 = div(n - 1, 1) + 1
                        __icse_27 = ((i_l_transformer_c1 - 1) * (__icse_25 * __icse_26) + (hh_transformer_c1 - 1) * __icse_26 + (i_transformer_c1 - 1)) + 1
                        __idx_row_max_transformer_c1_stack_6 = max(0, div(n_layers - 1, 1) + 1) * max(0, __icse_25) * max(0, __icse_26) * max(0, div(n - 2, 1) + 1) + __icse_27
                        row_max_transformer_c1_stack[__idx_row_max_transformer_c1_stack_6] = row_max_transformer_c1
                        __idx_row_sum_transformer_c1_stack_8 = __icse_27
                        row_sum_transformer_c1_stack[__idx_row_sum_transformer_c1_stack_8] = row_sum_transformer_c1
                    end
            end
            if div(n * dk - 1, 1) + 1 > 0
                JACC.@parallel_for range = div(n * dk - 1, 1) + 1 jacc_kernel_transformer_loss_b_8!(ctx, ctx_stack, d_transformer_c1, dk, h, head_offset_transformer_c1, hh_transformer_c1, i_l_transformer_c1, n, n_d_transformer_c1, n_layers, probs, s_transformer_c1_stack, score_off_transformer_c1, v)
            end
        end
        if div(n_d_transformer_c1 - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_b_9!(attn_out, b_offset_transformer_c1, bo, ctx, d_transformer_c1, dk, h, i_l_transformer_c1, n, n_d_transformer_c1, n_layers, s_transformer_c1_stack, w_offset_transformer_c1, wo)
        end
        if div(n_d_transformer_c1 - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_b_10!(attn_out, i_l_transformer_c1, n_d_transformer_c1, resid1, resid1_stack, x)
        end
        for i_transformer_c1 = 1:n
            s_transformer_c1 = 0.0
            for i_j_transformer_c1 = 1:d_transformer_c1
                CUDA.@allowscalar begin
                        s_transformer_c1 = s_transformer_c1 + resid1[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1]
                    end
            end
            CUDA.@allowscalar begin
                    __idx_row_mean_transformer_c1_stack_2 = ((i_l_transformer_c1 - 1) * (div(n - 1, 1) + 1) + (i_transformer_c1 - 1)) + 1
                    row_mean_transformer_c1_stack[__idx_row_mean_transformer_c1_stack_2] = row_mean_transformer_c1
                    row_mean_transformer_c1 = s_transformer_c1 / d_transformer_c1
                    s2_transformer_c1 = 0.0
                end
            for i_j_transformer_c1 = 1:d_transformer_c1
                CUDA.@allowscalar begin
                        diff_transformer_c1 = resid1[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1] - row_mean_transformer_c1
                        s2_transformer_c1 = s2_transformer_c1 + diff_transformer_c1 * diff_transformer_c1
                    end
            end
            CUDA.@allowscalar begin
                    __icse_39 = ((i_l_transformer_c1 - 1) * (div(n - 1, 1) + 1) + (i_transformer_c1 - 1)) + 1
                    __idx_row_var_transformer_c1_stack_7 = __icse_39
                    row_var_transformer_c1_stack[__idx_row_var_transformer_c1_stack_7] = row_var_transformer_c1
                    row_var_transformer_c1 = s2_transformer_c1 / d_transformer_c1
                    __idx_denom_transformer_c1_stack_10 = __icse_39
                    denom_transformer_c1_stack[__idx_denom_transformer_c1_stack_10] = denom_transformer_c1
                    denom_transformer_c1 = sqrt(row_var_transformer_c1 + eps)
                end
            if div(d_transformer_c1 - 1, 1) + 1 > 0
                JACC.@parallel_for range = div(d_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_b_11!(d_transformer_c1, denom_transformer_c1, i_l_transformer_c1, i_transformer_c1, ln1_bias, ln1_gain, ln_offset_transformer_c1, n, normed1, normed1_stack, resid1, row_mean_transformer_c1)
            end
            CUDA.@allowscalar begin
                    __icse_42 = ((i_l_transformer_c1 - 1) * (div(n - 1, 1) + 1) + (i_transformer_c1 - 1)) + 1
                    __idx_s2_transformer_c1_stack_14 = __icse_42
                    s2_transformer_c1_stack[__idx_s2_transformer_c1_stack_14] = s2_transformer_c1
                    __icse_43 = max(0, div(n_layers - 1, 1) + 1)
                    __icse_44 = __icse_43 * max(0, div(n_d_transformer_c1 - 1, 1) + 1)
                    __icse_45 = max(0, div(h - 1, 1) + 1)
                    __idx_s_transformer_c1_stack_16 = (((((__icse_44 + __icse_44) + __icse_44) + __icse_43 * __icse_45 * max(0, div(n * n - 1, 1) + 1)) + __icse_43 * __icse_45 * max(0, div(n * dk - 1, 1) + 1)) + __icse_44) + __icse_42
                    s_transformer_c1_stack[__idx_s_transformer_c1_stack_16] = s_transformer_c1
                end
        end
        if div(n_dff_transformer_c1 - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_dff_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_b_12!(b1, b1_offset_transformer_c1, d_transformer_c1, dff, dk, ff_hidden, ff_hidden_stack, h, i_l_transformer_c1, n, n_d_transformer_c1, n_dff_transformer_c1, n_layers, normed1, s_transformer_c1_stack, w1, w1_offset_transformer_c1)
        end
        if div(n_d_transformer_c1 - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_b_13!(b2, b2_offset_transformer_c1, d_transformer_c1, dff, dk, ff_hidden, ff_out, h, i_l_transformer_c1, n, n_d_transformer_c1, n_dff_transformer_c1, n_layers, s_transformer_c1_stack, w2, w2_offset_transformer_c1)
        end
        if div(n_d_transformer_c1 - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_b_14!(ff_out, i_l_transformer_c1, n_d_transformer_c1, normed1, resid2, resid2_stack)
        end
        for i_transformer_c1 = 1:n
            s_transformer_c1 = 0.0
            for i_j_transformer_c1 = 1:d_transformer_c1
                CUDA.@allowscalar begin
                        s_transformer_c1 = s_transformer_c1 + resid2[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1]
                    end
            end
            CUDA.@allowscalar begin
                    __icse_56 = div(n - 1, 1) + 1
                    __idx_row_mean_transformer_c1_stack_2 = max(0, div(n_layers - 1, 1) + 1) * max(0, __icse_56) + (((i_l_transformer_c1 - 1) * __icse_56 + (i_transformer_c1 - 1)) + 1)
                    row_mean_transformer_c1_stack[__idx_row_mean_transformer_c1_stack_2] = row_mean_transformer_c1
                    row_mean_transformer_c1 = s_transformer_c1 / d_transformer_c1
                    s2_transformer_c1 = 0.0
                end
            for i_j_transformer_c1 = 1:d_transformer_c1
                CUDA.@allowscalar begin
                        diff_transformer_c1 = resid2[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1] - row_mean_transformer_c1
                        s2_transformer_c1 = s2_transformer_c1 + diff_transformer_c1 * diff_transformer_c1
                    end
            end
            CUDA.@allowscalar begin
                    __icse_57 = div(n - 1, 1) + 1
                    __icse_58 = max(0, div(n_layers - 1, 1) + 1) * max(0, __icse_57) + (((i_l_transformer_c1 - 1) * __icse_57 + (i_transformer_c1 - 1)) + 1)
                    __idx_row_var_transformer_c1_stack_7 = __icse_58
                    row_var_transformer_c1_stack[__idx_row_var_transformer_c1_stack_7] = row_var_transformer_c1
                    row_var_transformer_c1 = s2_transformer_c1 / d_transformer_c1
                    __idx_denom_transformer_c1_stack_10 = __icse_58
                    denom_transformer_c1_stack[__idx_denom_transformer_c1_stack_10] = denom_transformer_c1
                    denom_transformer_c1 = sqrt(row_var_transformer_c1 + eps)
                end
            if div(d_transformer_c1 - 1, 1) + 1 > 0
                JACC.@parallel_for range = div(d_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_b_15!(d_transformer_c1, denom_transformer_c1, i_transformer_c1, ln2_bias, ln2_gain, ln2_offset_transformer_c1, resid2, row_mean_transformer_c1, x_next)
            end
            CUDA.@allowscalar begin
                    __icse_59 = max(0, div(n_layers - 1, 1) + 1)
                    __icse_60 = div(n - 1, 1) + 1
                    __icse_61 = __icse_59 * max(0, __icse_60)
                    __icse_62 = ((i_l_transformer_c1 - 1) * __icse_60 + (i_transformer_c1 - 1)) + 1
                    __idx_s2_transformer_c1_stack_14 = __icse_61 + __icse_62
                    s2_transformer_c1_stack[__idx_s2_transformer_c1_stack_14] = s2_transformer_c1
                    __icse_63 = __icse_59 * max(0, div(n_d_transformer_c1 - 1, 1) + 1)
                    __icse_64 = max(0, div(h - 1, 1) + 1)
                    __idx_s_transformer_c1_stack_16 = ((((((((__icse_63 + __icse_63) + __icse_63) + __icse_59 * __icse_64 * max(0, div(n * n - 1, 1) + 1)) + __icse_59 * __icse_64 * max(0, div(n * dk - 1, 1) + 1)) + __icse_63) + __icse_61) + __icse_59 * max(0, div(n_dff_transformer_c1 - 1, 1) + 1)) + __icse_63) + __icse_62
                    s_transformer_c1_stack[__idx_s_transformer_c1_stack_16] = s_transformer_c1
                end
        end
        if div(n_d_transformer_c1 - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_b_16!(i_l_transformer_c1, n_d_transformer_c1, x, x_next, x_stack)
        end
        CUDA.@allowscalar begin
                __icse_65 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)
                __icse_66 = (__icse_65 + __icse_65) + ((i_l_transformer_c1 - 1) + 1)
                __idx_denom_transformer_c1_stack_20 = __icse_66
                denom_transformer_c1_stack[__idx_denom_transformer_c1_stack_20] = denom_transformer_c1
                __idx_row_mean_transformer_c1_stack_22 = __icse_66
                row_mean_transformer_c1_stack[__idx_row_mean_transformer_c1_stack_22] = row_mean_transformer_c1
                __idx_row_var_transformer_c1_stack_24 = __icse_66
                row_var_transformer_c1_stack[__idx_row_var_transformer_c1_stack_24] = row_var_transformer_c1
            end
    end
    if div(n * h * dk - 1, 1) + 1 < 32768
        if div(n * h * dk - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n * h * dk - 1, 1) + 1 jacc_kernel_transformer_loss_b_18!(dk, h, loss, n, target, x)
        end
    else
        if div(n * h * dk - 1, 1) + 1 > 0
            __jgen_redval_17 = JACC.@parallel_reduce(range = div(n * h * dk - 1, 1) + 1, (((i_o, target, x)->(x[i_o] - target[i_o]) ^ 2))(target, x))
            JACC.@parallel_for range = 1 jacc_kernel_transformer_loss_b_17!(loss, __jgen_redval_17)
        end
    end
    CUDA.@allowscalar begin
            __icse_67 = max(0, div(n_layers - 1, 1) + 1)
            __icse_68 = __icse_67 * max(0, div(n - 1, 1) + 1)
            __icse_69 = ((__icse_68 + __icse_68) + __icse_67) + 1
            __idx_denom_transformer_c1_stack_7 = __icse_69
            denom_transformer_c1_stack[__idx_denom_transformer_c1_stack_7] = denom_transformer_c1
            __idx_row_mean_transformer_c1_stack_9 = __icse_69
            row_mean_transformer_c1_stack[__idx_row_mean_transformer_c1_stack_9] = row_mean_transformer_c1
            __idx_row_var_transformer_c1_stack_11 = __icse_69
            row_var_transformer_c1_stack[__idx_row_var_transformer_c1_stack_11] = row_var_transformer_c1
            __idx_denom_transformer_c1_stack_0 = __icse_69
            denom_transformer_c1 = denom_transformer_c1_stack[__idx_denom_transformer_c1_stack_0]
            __idx_row_mean_transformer_c1_stack_2 = __icse_69
            row_mean_transformer_c1 = row_mean_transformer_c1_stack[__idx_row_mean_transformer_c1_stack_2]
            __idx_row_var_transformer_c1_stack_4 = __icse_69
            row_var_transformer_c1 = row_var_transformer_c1_stack[__idx_row_var_transformer_c1_stack_4]
            d_transformer_c1 = h * dk
            n_d_transformer_c1 = n * d_transformer_c1
            n_dff_transformer_c1 = n * dff
        end
    if div(1 - n * h * dk, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n * h * dk, -1) + 1 jacc_kernel_transformer_loss_b_19!(dk, h, lossb, n, target, targetb, x, xb)
    end
    for i_l_transformer_c1 = n_layers:-1:1
        CUDA.@allowscalar begin
                __icse_71 = max(0, div(n_layers - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)
                __icse_72 = i_l_transformer_c1 - 1
                __icse_73 = (__icse_71 + __icse_71) + (__icse_72 + 1)
                __idx_denom_transformer_c1_stack_0 = __icse_73
                denom_transformer_c1 = denom_transformer_c1_stack[__idx_denom_transformer_c1_stack_0]
                __idx_row_mean_transformer_c1_stack_2 = __icse_73
                row_mean_transformer_c1 = row_mean_transformer_c1_stack[__idx_row_mean_transformer_c1_stack_2]
                __idx_row_var_transformer_c1_stack_4 = __icse_73
                row_var_transformer_c1 = row_var_transformer_c1_stack[__idx_row_var_transformer_c1_stack_4]
                w_offset_transformer_c1 = __icse_72 * d_transformer_c1 * d_transformer_c1
                __icse_74 = __icse_72 * d_transformer_c1
                b_offset_transformer_c1 = __icse_74
                ln_offset_transformer_c1 = __icse_74
                w1_offset_transformer_c1 = __icse_72 * d_transformer_c1 * dff
                b1_offset_transformer_c1 = __icse_72 * dff
                w2_offset_transformer_c1 = __icse_72 * dff * d_transformer_c1
                b2_offset_transformer_c1 = __icse_74
                ln2_offset_transformer_c1 = __icse_74
            end
        if div(1 - n_d_transformer_c1, -1) + 1 > 0
            JACC.@parallel_for range = div(1 - n_d_transformer_c1, -1) + 1 jacc_kernel_transformer_loss_b_20!(i_l_transformer_c1, n_d_transformer_c1, x, x_nextb, x_stack, xb)
        end
        for i_transformer_c1 = n:-1:1
            CUDA.@allowscalar begin
                    __icse_75 = max(0, div(n_layers - 1, 1) + 1)
                    __icse_76 = div(n - 1, 1) + 1
                    __icse_77 = __icse_75 * max(0, __icse_76)
                    __icse_78 = ((i_l_transformer_c1 - 1) * __icse_76 + (i_transformer_c1 - 1)) + 1
                    __idx_s2_transformer_c1_stack_0 = __icse_77 + __icse_78
                    s2_transformer_c1 = s2_transformer_c1_stack[__idx_s2_transformer_c1_stack_0]
                    __icse_79 = __icse_75 * max(0, div(n_d_transformer_c1 - 1, 1) + 1)
                    __icse_80 = max(0, div(h - 1, 1) + 1)
                    __idx_s_transformer_c1_stack_2 = ((((((((__icse_79 + __icse_79) + __icse_79) + __icse_75 * __icse_80 * max(0, div(n * n - 1, 1) + 1)) + __icse_75 * __icse_80 * max(0, div(n * dk - 1, 1) + 1)) + __icse_79) + __icse_77) + __icse_75 * max(0, div(n_dff_transformer_c1 - 1, 1) + 1)) + __icse_79) + __icse_78
                    s_transformer_c1 = s_transformer_c1_stack[__idx_s_transformer_c1_stack_2]
                end
            for j_transformer_c1 = d_transformer_c1:-1:1
                CUDA.@allowscalar begin
                        kk_transformer_c1 = (i_transformer_c1 - 1) * d_transformer_c1 + j_transformer_c1
                        __oldb_0 = x_nextb[kk_transformer_c1]
                        x_nextb[kk_transformer_c1] = 0.0
                        __cse_81 = ln2_gain[ln2_offset_transformer_c1 + j_transformer_c1] * __oldb_0
                        __cse_82 = (1.0 / denom_transformer_c1) * __cse_81
                        resid2b[kk_transformer_c1] = resid2b[kk_transformer_c1] + __cse_82
                        row_mean_transformer_c1b = row_mean_transformer_c1b + -__cse_82
                        __cse_83 = resid2[kk_transformer_c1] - row_mean_transformer_c1
                        denom_transformer_c1b = denom_transformer_c1b + -(__cse_83 / denom_transformer_c1 ^ 2) * __cse_81
                        ln2_gainb[ln2_offset_transformer_c1 + j_transformer_c1] = ln2_gainb[ln2_offset_transformer_c1 + j_transformer_c1] + (__cse_83 / denom_transformer_c1) * __oldb_0
                        ln2_biasb[ln2_offset_transformer_c1 + j_transformer_c1] = ln2_biasb[ln2_offset_transformer_c1 + j_transformer_c1] + __oldb_0
                    end
            end
            CUDA.@allowscalar begin
                    __icse_84 = div(n - 1, 1) + 1
                    __icse_85 = max(0, div(n_layers - 1, 1) + 1) * max(0, __icse_84) + (((i_l_transformer_c1 - 1) * __icse_84 + (i_transformer_c1 - 1)) + 1)
                    __idx_denom_transformer_c1_stack_0 = __icse_85
                    denom_transformer_c1 = denom_transformer_c1_stack[__idx_denom_transformer_c1_stack_0]
                    __oldb_2 = denom_transformer_c1b
                    denom_transformer_c1b = 0.0
                    __cse_86 = (1.0 / (2.0 * sqrt(row_var_transformer_c1 + eps))) * __oldb_2
                    row_var_transformer_c1b = row_var_transformer_c1b + __cse_86
                    epsb = epsb + __cse_86
                    __idx_row_var_transformer_c1_stack_0 = __icse_85
                    row_var_transformer_c1 = row_var_transformer_c1_stack[__idx_row_var_transformer_c1_stack_0]
                    __oldb_2 = row_var_transformer_c1b
                    row_var_transformer_c1b = 0.0
                    s2_transformer_c1b = s2_transformer_c1b + (1.0 / d_transformer_c1) * __oldb_2
                end
            for i_j_transformer_c1 = 1:d_transformer_c1
                CUDA.@allowscalar begin
                        diff_transformer_c1 = resid2[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1] - row_mean_transformer_c1
                        s2_transformer_c1 = s2_transformer_c1 + diff_transformer_c1 * diff_transformer_c1
                        __cse_87 = diff_transformer_c1 * s2_transformer_c1b
                        diff_transformer_c1b = diff_transformer_c1b + __cse_87
                        diff_transformer_c1b = diff_transformer_c1b + __cse_87
                        __oldb_0 = diff_transformer_c1b
                        diff_transformer_c1b = 0.0
                        resid2b[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1] = resid2b[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1] + __oldb_0
                        row_mean_transformer_c1b = row_mean_transformer_c1b + -__oldb_0
                    end
            end
            CUDA.@allowscalar begin
                    s2_transformer_c1b = 0.0
                    __icse_88 = div(n - 1, 1) + 1
                    __idx_row_mean_transformer_c1_stack_0 = max(0, div(n_layers - 1, 1) + 1) * max(0, __icse_88) + (((i_l_transformer_c1 - 1) * __icse_88 + (i_transformer_c1 - 1)) + 1)
                    row_mean_transformer_c1 = row_mean_transformer_c1_stack[__idx_row_mean_transformer_c1_stack_0]
                    __oldb_2 = row_mean_transformer_c1b
                    row_mean_transformer_c1b = 0.0
                    s_transformer_c1b = s_transformer_c1b + (1.0 / d_transformer_c1) * __oldb_2
                end
            for i_j_transformer_c1 = 1:d_transformer_c1
                CUDA.@allowscalar begin
                        s_transformer_c1 = s_transformer_c1 + resid2[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1]
                        resid2b[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1] = resid2b[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1] + s_transformer_c1b
                    end
            end
            s_transformer_c1b = 0.0
        end
        if div(1 - n_d_transformer_c1, -1) + 1 > 0
            JACC.@parallel_for range = div(1 - n_d_transformer_c1, -1) + 1 jacc_kernel_transformer_loss_b_21!(ff_outb, i_l_transformer_c1, n_d_transformer_c1, normed1b, resid2, resid2_stack, resid2b)
        end
        if div(1 - n_d_transformer_c1, -1) + 1 > 0
            JACC.@parallel_for range = div(1 - n_d_transformer_c1, -1) + 1 jacc_kernel_transformer_loss_b_22!(b2_offset_transformer_c1, b2b, d_transformer_c1, dff, dk, ff_hidden, ff_hiddenb, ff_outb, h, i_l_transformer_c1, n, n_d_transformer_c1, n_dff_transformer_c1, n_layers, s_transformer_c1_stack, w2, w2_offset_transformer_c1, w2b)
        end
        if div(1 - n_dff_transformer_c1, -1) + 1 > 0
            JACC.@parallel_for range = div(1 - n_dff_transformer_c1, -1) + 1 jacc_kernel_transformer_loss_b_23!(b1, b1_offset_transformer_c1, b1b, d_transformer_c1, dff, dk, ff_hidden, ff_hidden_stack, ff_hiddenb, h, i_l_transformer_c1, n, n_d_transformer_c1, n_dff_transformer_c1, n_layers, normed1, normed1b, s_transformer_c1_stack, w1, w1_offset_transformer_c1, w1b)
        end
        for i_transformer_c1 = n:-1:1
            CUDA.@allowscalar begin
                    __icse_104 = ((i_l_transformer_c1 - 1) * (div(n - 1, 1) + 1) + (i_transformer_c1 - 1)) + 1
                    __idx_s2_transformer_c1_stack_0 = __icse_104
                    s2_transformer_c1 = s2_transformer_c1_stack[__idx_s2_transformer_c1_stack_0]
                    __icse_105 = max(0, div(n_layers - 1, 1) + 1)
                    __icse_106 = __icse_105 * max(0, div(n_d_transformer_c1 - 1, 1) + 1)
                    __icse_107 = max(0, div(h - 1, 1) + 1)
                    __idx_s_transformer_c1_stack_2 = (((((__icse_106 + __icse_106) + __icse_106) + __icse_105 * __icse_107 * max(0, div(n * n - 1, 1) + 1)) + __icse_105 * __icse_107 * max(0, div(n * dk - 1, 1) + 1)) + __icse_106) + __icse_104
                    s_transformer_c1 = s_transformer_c1_stack[__idx_s_transformer_c1_stack_2]
                end
            for j_transformer_c1 = d_transformer_c1:-1:1
                CUDA.@allowscalar begin
                        __icse_108 = i_transformer_c1 - 1
                        kk_transformer_c1 = __icse_108 * d_transformer_c1 + j_transformer_c1
                        __icse_109 = div(d_transformer_c1 - 1, 1) + 1
                        __idx_normed1_stack_0 = ((i_l_transformer_c1 - 1) * ((div(n - 1, 1) + 1) * __icse_109) + __icse_108 * __icse_109 + (j_transformer_c1 - 1)) + 1
                        normed1[kk_transformer_c1] = normed1_stack[__idx_normed1_stack_0]
                        __oldb_2 = normed1b[kk_transformer_c1]
                        normed1b[kk_transformer_c1] = 0.0
                        __cse_110 = ln1_gain[ln_offset_transformer_c1 + j_transformer_c1] * __oldb_2
                        __cse_111 = (1.0 / denom_transformer_c1) * __cse_110
                        resid1b[kk_transformer_c1] = resid1b[kk_transformer_c1] + __cse_111
                        row_mean_transformer_c1b = row_mean_transformer_c1b + -__cse_111
                        __cse_112 = resid1[kk_transformer_c1] - row_mean_transformer_c1
                        denom_transformer_c1b = denom_transformer_c1b + -(__cse_112 / denom_transformer_c1 ^ 2) * __cse_110
                        ln1_gainb[ln_offset_transformer_c1 + j_transformer_c1] = ln1_gainb[ln_offset_transformer_c1 + j_transformer_c1] + (__cse_112 / denom_transformer_c1) * __oldb_2
                        ln1_biasb[ln_offset_transformer_c1 + j_transformer_c1] = ln1_biasb[ln_offset_transformer_c1 + j_transformer_c1] + __oldb_2
                    end
            end
            CUDA.@allowscalar begin
                    __icse_113 = ((i_l_transformer_c1 - 1) * (div(n - 1, 1) + 1) + (i_transformer_c1 - 1)) + 1
                    __idx_denom_transformer_c1_stack_0 = __icse_113
                    denom_transformer_c1 = denom_transformer_c1_stack[__idx_denom_transformer_c1_stack_0]
                    __oldb_2 = denom_transformer_c1b
                    denom_transformer_c1b = 0.0
                    __cse_114 = (1.0 / (2.0 * sqrt(row_var_transformer_c1 + eps))) * __oldb_2
                    row_var_transformer_c1b = row_var_transformer_c1b + __cse_114
                    epsb = epsb + __cse_114
                    __idx_row_var_transformer_c1_stack_0 = __icse_113
                    row_var_transformer_c1 = row_var_transformer_c1_stack[__idx_row_var_transformer_c1_stack_0]
                    __oldb_2 = row_var_transformer_c1b
                    row_var_transformer_c1b = 0.0
                    s2_transformer_c1b = s2_transformer_c1b + (1.0 / d_transformer_c1) * __oldb_2
                end
            for i_j_transformer_c1 = 1:d_transformer_c1
                CUDA.@allowscalar begin
                        diff_transformer_c1 = resid1[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1] - row_mean_transformer_c1
                        s2_transformer_c1 = s2_transformer_c1 + diff_transformer_c1 * diff_transformer_c1
                        __cse_115 = diff_transformer_c1 * s2_transformer_c1b
                        diff_transformer_c1b = diff_transformer_c1b + __cse_115
                        diff_transformer_c1b = diff_transformer_c1b + __cse_115
                        __oldb_0 = diff_transformer_c1b
                        diff_transformer_c1b = 0.0
                        resid1b[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1] = resid1b[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1] + __oldb_0
                        row_mean_transformer_c1b = row_mean_transformer_c1b + -__oldb_0
                    end
            end
            CUDA.@allowscalar begin
                    s2_transformer_c1b = 0.0
                    __idx_row_mean_transformer_c1_stack_0 = ((i_l_transformer_c1 - 1) * (div(n - 1, 1) + 1) + (i_transformer_c1 - 1)) + 1
                    row_mean_transformer_c1 = row_mean_transformer_c1_stack[__idx_row_mean_transformer_c1_stack_0]
                    __oldb_2 = row_mean_transformer_c1b
                    row_mean_transformer_c1b = 0.0
                    s_transformer_c1b = s_transformer_c1b + (1.0 / d_transformer_c1) * __oldb_2
                end
            for i_j_transformer_c1 = 1:d_transformer_c1
                CUDA.@allowscalar begin
                        s_transformer_c1 = s_transformer_c1 + resid1[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1]
                        resid1b[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1] = resid1b[(i_transformer_c1 - 1) * d_transformer_c1 + i_j_transformer_c1] + s_transformer_c1b
                    end
            end
            s_transformer_c1b = 0.0
        end
        if div(1 - n_d_transformer_c1, -1) + 1 > 0
            JACC.@parallel_for range = div(1 - n_d_transformer_c1, -1) + 1 jacc_kernel_transformer_loss_b_24!(attn_outb, i_l_transformer_c1, n_d_transformer_c1, resid1, resid1_stack, resid1b, xb)
        end
        if div(1 - n_d_transformer_c1, -1) + 1 > 0
            JACC.@parallel_for range = div(1 - n_d_transformer_c1, -1) + 1 jacc_kernel_transformer_loss_b_25!(attn_outb, b_offset_transformer_c1, bob, ctx, ctxb, d_transformer_c1, dk, h, i_l_transformer_c1, n, n_d_transformer_c1, n_layers, s_transformer_c1_stack, w_offset_transformer_c1, wo, wob)
        end
        for hh_transformer_c1 = h:-1:1
            __icse_123 = hh_transformer_c1 - 1
            head_offset_transformer_c1 = __icse_123 * dk
            score_off_transformer_c1 = __icse_123 * n * n
            if div(1 - n * dk, -1) + 1 > 0
                JACC.@parallel_for range = div(1 - n * dk, -1) + 1 jacc_kernel_transformer_loss_b_26!(ctx, ctx_stack, ctxb, d_transformer_c1, dk, h, head_offset_transformer_c1, hh_transformer_c1, i_l_transformer_c1, n, n_d_transformer_c1, n_layers, probs, probsb, s_transformer_c1_stack, score_off_transformer_c1, v, vb)
            end
            for i_transformer_c1 = n:-1:1
                CUDA.@allowscalar begin
                        __icse_132 = div(h - 1, 1) + 1
                        __icse_133 = div(n - 1, 1) + 1
                        __icse_134 = ((i_l_transformer_c1 - 1) * (__icse_132 * __icse_133) + (hh_transformer_c1 - 1) * __icse_133 + (i_transformer_c1 - 1)) + 1
                        __idx_row_max_transformer_c1_stack_0 = max(0, div(n_layers - 1, 1) + 1) * max(0, __icse_132) * max(0, __icse_133) * max(0, div(n - 2, 1) + 1) + __icse_134
                        row_max_transformer_c1 = row_max_transformer_c1_stack[__idx_row_max_transformer_c1_stack_0]
                        __idx_row_sum_transformer_c1_stack_2 = __icse_134
                        row_sum_transformer_c1 = row_sum_transformer_c1_stack[__idx_row_sum_transformer_c1_stack_2]
                    end
                for j_transformer_c1 = n:-1:1
                    CUDA.@allowscalar begin
                            __icse_135 = i_transformer_c1 - 1
                            kk_transformer_c1 = score_off_transformer_c1 + __icse_135 * n + j_transformer_c1
                            __icse_136 = div(h - 1, 1) + 1
                            __icse_137 = div(n - 1, 1) + 1
                            __icse_138 = max(0, __icse_137)
                            __idx_probs_stack_0 = max(0, div(n_layers - 1, 1) + 1) * max(0, __icse_136) * __icse_138 * __icse_138 + (((i_l_transformer_c1 - 1) * (__icse_136 * __icse_137 * __icse_137) + (hh_transformer_c1 - 1) * (__icse_137 * __icse_137) + __icse_135 * __icse_137 + (j_transformer_c1 - 1)) + 1)
                            probs[kk_transformer_c1] = probs_stack[__idx_probs_stack_0]
                            __oldb_2 = probsb[kk_transformer_c1]
                            probsb[kk_transformer_c1] = 0.0
                            probsb[kk_transformer_c1] = probsb[kk_transformer_c1] + (1.0 / row_sum_transformer_c1) * __oldb_2
                            row_sum_transformer_c1b = row_sum_transformer_c1b + -(probs[kk_transformer_c1] / row_sum_transformer_c1 ^ 2) * __oldb_2
                        end
                end
                for i_j_transformer_c1 = 1:n
                    CUDA.@allowscalar begin
                            row_sum_transformer_c1 = row_sum_transformer_c1 + probs[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + i_j_transformer_c1]
                            probsb[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + i_j_transformer_c1] = probsb[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + i_j_transformer_c1] + row_sum_transformer_c1b
                        end
                end
                row_sum_transformer_c1b = 0.0
                for j_transformer_c1 = n:-1:1
                    CUDA.@allowscalar begin
                            __icse_139 = i_transformer_c1 - 1
                            kk_transformer_c1 = score_off_transformer_c1 + __icse_139 * n + j_transformer_c1
                            __icse_140 = div(n - 1, 1) + 1
                            __idx_probs_stack_0 = ((i_l_transformer_c1 - 1) * ((div(h - 1, 1) + 1) * __icse_140 * __icse_140) + (hh_transformer_c1 - 1) * (__icse_140 * __icse_140) + __icse_139 * __icse_140 + (j_transformer_c1 - 1)) + 1
                            probs[kk_transformer_c1] = probs_stack[__idx_probs_stack_0]
                            __oldb_2 = probsb[kk_transformer_c1]
                            probsb[kk_transformer_c1] = 0.0
                            __cse_141 = exp(scores[kk_transformer_c1] - row_max_transformer_c1) * __oldb_2
                            scoresb[kk_transformer_c1] = scoresb[kk_transformer_c1] + __cse_141
                            row_max_transformer_c1b = row_max_transformer_c1b + -__cse_141
                        end
                end
                for i_j_transformer_c1 = n:-1:2
                    CUDA.@allowscalar begin
                            __icse_142 = div(n - 1, 1) + 1
                            __icse_143 = div(n - 2, 1) + 1
                            __idx_row_max_transformer_c1_stack_0 = ((i_l_transformer_c1 - 1) * ((div(h - 1, 1) + 1) * __icse_142 * __icse_143) + (hh_transformer_c1 - 1) * (__icse_142 * __icse_143) + (i_transformer_c1 - 1) * __icse_143 + (i_j_transformer_c1 - 2)) + 1
                            row_max_transformer_c1 = row_max_transformer_c1_stack[__idx_row_max_transformer_c1_stack_0]
                            __oldb_2 = row_max_transformer_c1b
                            row_max_transformer_c1b = 0.0
                            __cse_144 = scores[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + i_j_transformer_c1]
                            row_max_transformer_c1b = row_max_transformer_c1b + (0.5 * (1.0 + sign(row_max_transformer_c1 - __cse_144))) * __oldb_2
                            scoresb[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + i_j_transformer_c1] = scoresb[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + i_j_transformer_c1] + (0.5 * (1.0 + sign(__cse_144 - row_max_transformer_c1))) * __oldb_2
                        end
                end
                CUDA.@allowscalar begin
                        __oldb_0 = row_max_transformer_c1b
                        row_max_transformer_c1b = 0.0
                        scoresb[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + 1] = scoresb[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + 1] + __oldb_0
                    end
            end
            if div(1 - n * n, -1) + 1 > 0
                JACC.@parallel_for range = div(1 - n * n, -1) + 1 jacc_kernel_transformer_loss_b_27!(d_transformer_c1, dk, h, head_offset_transformer_c1, hh_transformer_c1, i_l_transformer_c1, inv_sqrt_dk_transformer_c1, k, kb, n, n_d_transformer_c1, n_layers, q, qb, s_transformer_c1_stack, score_off_transformer_c1, scores, scores_stack, scoresb)
            end
        end
        if div(1 - n_d_transformer_c1, -1) + 1 > 0
            JACC.@parallel_for range = div(1 - n_d_transformer_c1, -1) + 1 jacc_kernel_transformer_loss_b_28!(b_offset_transformer_c1, bvb, d_transformer_c1, i_l_transformer_c1, n_d_transformer_c1, n_layers, s_transformer_c1_stack, v, v_stack, vb, w_offset_transformer_c1, wv, wvb, x, xb)
        end
        if div(1 - n_d_transformer_c1, -1) + 1 > 0
            JACC.@parallel_for range = div(1 - n_d_transformer_c1, -1) + 1 jacc_kernel_transformer_loss_b_29!(b_offset_transformer_c1, bkb, d_transformer_c1, i_l_transformer_c1, k, k_stack, kb, n_d_transformer_c1, n_layers, s_transformer_c1_stack, w_offset_transformer_c1, wk, wkb, x, xb)
        end
        if div(1 - n_d_transformer_c1, -1) + 1 > 0
            JACC.@parallel_for range = div(1 - n_d_transformer_c1, -1) + 1 jacc_kernel_transformer_loss_b_30!(b_offset_transformer_c1, bqb, d_transformer_c1, i_l_transformer_c1, n_d_transformer_c1, q, q_stack, qb, s_transformer_c1_stack, w_offset_transformer_c1, wq, wqb, x, xb)
        end
    end
    inv_sqrt_dk_transformer_c1b = 0.0
    if div(1 - n * h * dk, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n * h * dk, -1) + 1 jacc_kernel_transformer_loss_b_31!(dk, h, n, x_inb, xb)
    end
    return epsb
end

function transformer_loss_jacc(x, wq, bq, wk, bk, wv, bv, wo, bo, ln1_gain, ln1_bias, w1, b1, w2, b2, ln2_gain, ln2_bias, q, k, v, scores, probs, ctx, attn_out, resid1, normed1, ff_hidden, ff_out, resid2, x_next, n, dk, h, dff, n_layers, eps, x_in, target, loss)
    if div(n * h * dk - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n * h * dk - 1, 1) + 1 jacc_kernel_transformer_loss_1!(dk, h, n, x, x_in)
    end
    d_transformer_c1 = h * dk
    inv_sqrt_dk_transformer_c1 = 1.0 / sqrt(dk)
    n_d_transformer_c1 = n * d_transformer_c1
    n_dff_transformer_c1 = n * dff
    for i_l_transformer_c1 = 1:n_layers
        w_offset_transformer_c1 = (i_l_transformer_c1 - 1) * d_transformer_c1 * d_transformer_c1
        b_offset_transformer_c1 = (i_l_transformer_c1 - 1) * d_transformer_c1
        ln_offset_transformer_c1 = (i_l_transformer_c1 - 1) * d_transformer_c1
        w1_offset_transformer_c1 = (i_l_transformer_c1 - 1) * d_transformer_c1 * dff
        b1_offset_transformer_c1 = (i_l_transformer_c1 - 1) * dff
        w2_offset_transformer_c1 = (i_l_transformer_c1 - 1) * dff * d_transformer_c1
        b2_offset_transformer_c1 = (i_l_transformer_c1 - 1) * d_transformer_c1
        ln2_offset_transformer_c1 = (i_l_transformer_c1 - 1) * d_transformer_c1
        if div(n_d_transformer_c1 - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_2!(b_offset_transformer_c1, bq, d_transformer_c1, n_d_transformer_c1, q, w_offset_transformer_c1, wq, x)
        end
        if div(n_d_transformer_c1 - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_3!(b_offset_transformer_c1, bk, d_transformer_c1, k, n_d_transformer_c1, w_offset_transformer_c1, wk, x)
        end
        if div(n_d_transformer_c1 - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_4!(b_offset_transformer_c1, bv, d_transformer_c1, n_d_transformer_c1, v, w_offset_transformer_c1, wv, x)
        end
        for hh_transformer_c1 = 1:h
            head_offset_transformer_c1 = (hh_transformer_c1 - 1) * dk
            score_off_transformer_c1 = (hh_transformer_c1 - 1) * n * n
            if div(n * n - 1, 1) + 1 > 0
                JACC.@parallel_for range = div(n * n - 1, 1) + 1 jacc_kernel_transformer_loss_5!(d_transformer_c1, dk, head_offset_transformer_c1, inv_sqrt_dk_transformer_c1, k, n, q, score_off_transformer_c1, scores)
            end
            for i_transformer_c1 = 1:n
                CUDA.@allowscalar begin
                        row_max_transformer_c1 = scores[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + 1]
                    end
                for i_j_transformer_c1 = 2:n
                    CUDA.@allowscalar begin
                            row_max_transformer_c1 = max(row_max_transformer_c1, scores[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + i_j_transformer_c1])
                        end
                end
                if div(n - 1, 1) + 1 > 0
                    JACC.@parallel_for range = div(n - 1, 1) + 1 jacc_kernel_transformer_loss_6!(i_transformer_c1, n, probs, row_max_transformer_c1, score_off_transformer_c1, scores)
                end
                row_sum_transformer_c1 = 0.0
                for i_j_transformer_c1 = 1:n
                    CUDA.@allowscalar begin
                            row_sum_transformer_c1 = row_sum_transformer_c1 + probs[score_off_transformer_c1 + (i_transformer_c1 - 1) * n + i_j_transformer_c1]
                        end
                end
                if div(n - 1, 1) + 1 > 0
                    JACC.@parallel_for range = div(n - 1, 1) + 1 jacc_kernel_transformer_loss_7!(i_transformer_c1, n, probs, row_sum_transformer_c1, score_off_transformer_c1)
                end
            end
            if div(n * dk - 1, 1) + 1 > 0
                JACC.@parallel_for range = div(n * dk - 1, 1) + 1 jacc_kernel_transformer_loss_8!(ctx, d_transformer_c1, dk, head_offset_transformer_c1, n, probs, score_off_transformer_c1, v)
            end
        end
        if div(n_d_transformer_c1 - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_9!(attn_out, b_offset_transformer_c1, bo, ctx, d_transformer_c1, n_d_transformer_c1, w_offset_transformer_c1, wo)
        end
        if div(n_d_transformer_c1 - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_10!(attn_out, n_d_transformer_c1, resid1, x)
        end
        if div(n - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n - 1, 1) + 1 jacc_kernel_transformer_loss_11!(d_transformer_c1, eps, ln1_bias, ln1_gain, ln_offset_transformer_c1, n, normed1, resid1)
        end
        if div(n_dff_transformer_c1 - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_dff_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_12!(b1, b1_offset_transformer_c1, d_transformer_c1, dff, ff_hidden, n_dff_transformer_c1, normed1, w1, w1_offset_transformer_c1)
        end
        if div(n_d_transformer_c1 - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_13!(b2, b2_offset_transformer_c1, d_transformer_c1, dff, ff_hidden, ff_out, n_d_transformer_c1, w2, w2_offset_transformer_c1)
        end
        if div(n_d_transformer_c1 - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_14!(ff_out, n_d_transformer_c1, normed1, resid2)
        end
        if div(n - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n - 1, 1) + 1 jacc_kernel_transformer_loss_15!(d_transformer_c1, eps, ln2_bias, ln2_gain, ln2_offset_transformer_c1, n, resid2, x_next)
        end
        if div(n_d_transformer_c1 - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_d_transformer_c1 - 1, 1) + 1 jacc_kernel_transformer_loss_16!(n_d_transformer_c1, x, x_next)
        end
    end
    if div(n * h * dk - 1, 1) + 1 < 32768
        if div(n * h * dk - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n * h * dk - 1, 1) + 1 jacc_kernel_transformer_loss_18!(dk, h, loss, n, target, x)
        end
    else
        if div(n * h * dk - 1, 1) + 1 > 0
            __jgen_redval_17 = JACC.@parallel_reduce(range = div(n * h * dk - 1, 1) + 1, (((i_o, target, x)->(x[i_o] - target[i_o]) ^ 2))(target, x))
            JACC.@parallel_for range = 1 jacc_kernel_transformer_loss_17!(loss, __jgen_redval_17)
        end
    end
    return nothing
end
