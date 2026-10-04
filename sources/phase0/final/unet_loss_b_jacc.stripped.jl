using CUDA
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_unet_loss_b_1!(__jacc_i, n_xpad0_unet_c1, xpad0, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    xpad0[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_b_2!(__jacc_i, c_in, hp1_unet_c1, hw_unet_c1, pad_unet_c1, w, wp1_unet_c1, x, xpad0)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp1_unet_c1 * wp1_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp1_unet_c1 + (j_unet_c1 + pad_unet_c1)
    xpad0[yi_unet_c1] = x[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_3!(__jacc_i, b_e1a, c1, c_in, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_e1, w, w_e1a, wp1_unet_c1, xpad0)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c_in * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_5 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_5 * hp1_unet_c1 * wp1_unet_c1 + (row_unet_c1 - 1) * wp1_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c_in + __icse_5) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + xpad0[xi_unet_c1] * w_e1a[wi_unet_c1]
    end
    t_e1[idx_unet_c1] = s_unet_c1 + b_e1a[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_4!(__jacc_i, n_e1_mid_unet_c1, t_e1, t_e1_stack, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    __idx_t_e1_stack_0 = (i_unet_c1 - 1) + 1
    __cse_6 = t_e1[i_unet_c1]
    t_e1_stack[__idx_t_e1_stack_0] = __cse_6
    t_e1[i_unet_c1] = max(__cse_6, zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_b_5!(__jacc_i, n_e1_midpad_unet_c1, t_e1pad, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    t_e1pad[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_b_6!(__jacc_i, c1, hp1_unet_c1, hw_unet_c1, pad_unet_c1, t_e1, t_e1pad, w, wp1_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp1_unet_c1 * wp1_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp1_unet_c1 + (j_unet_c1 + pad_unet_c1)
    t_e1pad[yi_unet_c1] = t_e1[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_7!(__jacc_i, b_e1b, c1, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, skip1, t_e1pad, w, w_e1b, wp1_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c1 * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_7 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_7 * hp1_unet_c1 * wp1_unet_c1 + (row_unet_c1 - 1) * wp1_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c1 + __icse_7) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + t_e1pad[xi_unet_c1] * w_e1b[wi_unet_c1]
    end
    skip1[idx_unet_c1] = s_unet_c1 + b_e1b[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_8!(__jacc_i, n_e1_out_unet_c1, skip1, skip1_stack, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    __idx_skip1_stack_0 = (i_unet_c1 - 1) + 1
    __cse_8 = skip1[i_unet_c1]
    skip1_stack[__idx_skip1_stack_0] = __cse_8
    skip1[i_unet_c1] = max(__cse_8, zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_b_9!(__jacc_i, c1, hw2_unet_c1, hw_unet_c1, p1, skip1, w, w2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    oi_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    oj_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    i0_unet_c1 = 2oi_unet_c1 - 1
    j0_unet_c1 = 2oj_unet_c1 - 1
    a11_unet_c1 = skip1[(ci_unet_c1 - 1) * hw_unet_c1 + (i0_unet_c1 - 1) * w + j0_unet_c1]
    a12_unet_c1 = skip1[(ci_unet_c1 - 1) * hw_unet_c1 + (i0_unet_c1 - 1) * w + j0_unet_c1 + 1]
    a21_unet_c1 = skip1[(ci_unet_c1 - 1) * hw_unet_c1 + i0_unet_c1 * w + j0_unet_c1]
    a22_unet_c1 = skip1[(ci_unet_c1 - 1) * hw_unet_c1 + i0_unet_c1 * w + j0_unet_c1 + 1]
    m1_unet_c1 = max(a11_unet_c1, a12_unet_c1)
    m2_unet_c1 = max(a21_unet_c1, a22_unet_c1)
    p1[idx_unet_c1] = max(m1_unet_c1, m2_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_b_10!(__jacc_i, n_p1pad_unet_c1, p1pad, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    p1pad[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_b_11!(__jacc_i, c1, hp2_unet_c1, hw2_unet_c1, p1, p1pad, pad_unet_c1, w2_unet_c1, wp2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp2_unet_c1 * wp2_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp2_unet_c1 + (j_unet_c1 + pad_unet_c1)
    p1pad[yi_unet_c1] = p1[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_12!(__jacc_i, b_e2a, c1, c2, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, p1pad, t_e2, w2_unet_c1, w_e2a, wp2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c1 * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_9 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_9 * hp2_unet_c1 * wp2_unet_c1 + (row_unet_c1 - 1) * wp2_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c1 + __icse_9) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + p1pad[xi_unet_c1] * w_e2a[wi_unet_c1]
    end
    t_e2[idx_unet_c1] = s_unet_c1 + b_e2a[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_13!(__jacc_i, n_e2_mid_unet_c1, t_e2, t_e2_stack, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    __idx_t_e2_stack_0 = (i_unet_c1 - 1) + 1
    __cse_10 = t_e2[i_unet_c1]
    t_e2_stack[__idx_t_e2_stack_0] = __cse_10
    t_e2[i_unet_c1] = max(__cse_10, zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_b_14!(__jacc_i, n_e2_midpad_unet_c1, t_e2pad, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    t_e2pad[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_b_15!(__jacc_i, c2, hp2_unet_c1, hw2_unet_c1, pad_unet_c1, t_e2, t_e2pad, w2_unet_c1, wp2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp2_unet_c1 * wp2_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp2_unet_c1 + (j_unet_c1 + pad_unet_c1)
    t_e2pad[yi_unet_c1] = t_e2[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_16!(__jacc_i, b_e2b, c2, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, skip2, t_e2pad, w2_unet_c1, w_e2b, wp2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c2 * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_11 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_11 * hp2_unet_c1 * wp2_unet_c1 + (row_unet_c1 - 1) * wp2_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c2 + __icse_11) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + t_e2pad[xi_unet_c1] * w_e2b[wi_unet_c1]
    end
    skip2[idx_unet_c1] = s_unet_c1 + b_e2b[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_17!(__jacc_i, n_e2_out_unet_c1, skip2, skip2_stack, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    __idx_skip2_stack_0 = (i_unet_c1 - 1) + 1
    __cse_12 = skip2[i_unet_c1]
    skip2_stack[__idx_skip2_stack_0] = __cse_12
    skip2[i_unet_c1] = max(__cse_12, zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_b_18!(__jacc_i, c2, hw2_unet_c1, hw4_unet_c1, p2, skip2, w2_unet_c1, w4_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw4_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw4_unet_c1)
    oi_unet_c1 = div(rem_unet_c1, w4_unet_c1) + 1
    oj_unet_c1 = mod(rem_unet_c1, w4_unet_c1) + 1
    i0_unet_c1 = 2oi_unet_c1 - 1
    j0_unet_c1 = 2oj_unet_c1 - 1
    a11_unet_c1 = skip2[(ci_unet_c1 - 1) * hw2_unet_c1 + (i0_unet_c1 - 1) * w2_unet_c1 + j0_unet_c1]
    a12_unet_c1 = skip2[(ci_unet_c1 - 1) * hw2_unet_c1 + (i0_unet_c1 - 1) * w2_unet_c1 + j0_unet_c1 + 1]
    a21_unet_c1 = skip2[(ci_unet_c1 - 1) * hw2_unet_c1 + i0_unet_c1 * w2_unet_c1 + j0_unet_c1]
    a22_unet_c1 = skip2[(ci_unet_c1 - 1) * hw2_unet_c1 + i0_unet_c1 * w2_unet_c1 + j0_unet_c1 + 1]
    m1_unet_c1 = max(a11_unet_c1, a12_unet_c1)
    m2_unet_c1 = max(a21_unet_c1, a22_unet_c1)
    p2[idx_unet_c1] = max(m1_unet_c1, m2_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_b_19!(__jacc_i, n_p2pad_unet_c1, p2pad, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    p2pad[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_b_20!(__jacc_i, c2, hp4_unet_c1, hw4_unet_c1, p2, p2pad, pad_unet_c1, w4_unet_c1, wp4_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw4_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw4_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w4_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w4_unet_c1) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp4_unet_c1 * wp4_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp4_unet_c1 + (j_unet_c1 + pad_unet_c1)
    p2pad[yi_unet_c1] = p2[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_21!(__jacc_i, b_ba, c2, c3, hp4_unet_c1, hw4_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, p2pad, t_b, w4_unet_c1, w_ba, wp4_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw4_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw4_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w4_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w4_unet_c1) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c2 * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_13 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_13 * hp4_unet_c1 * wp4_unet_c1 + (row_unet_c1 - 1) * wp4_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c2 + __icse_13) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + p2pad[xi_unet_c1] * w_ba[wi_unet_c1]
    end
    t_b[idx_unet_c1] = s_unet_c1 + b_ba[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_22!(__jacc_i, n_b_mid_unet_c1, t_b, t_b_stack, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    __idx_t_b_stack_0 = (i_unet_c1 - 1) + 1
    __cse_14 = t_b[i_unet_c1]
    t_b_stack[__idx_t_b_stack_0] = __cse_14
    t_b[i_unet_c1] = max(__cse_14, zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_b_23!(__jacc_i, n_b_midpad_unet_c1, t_bpad, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    t_bpad[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_b_24!(__jacc_i, c3, hp4_unet_c1, hw4_unet_c1, pad_unet_c1, t_b, t_bpad, w4_unet_c1, wp4_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw4_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw4_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w4_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w4_unet_c1) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp4_unet_c1 * wp4_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp4_unet_c1 + (j_unet_c1 + pad_unet_c1)
    t_bpad[yi_unet_c1] = t_b[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_25!(__jacc_i, b_bb, bott, c3, hp4_unet_c1, hw4_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_bpad, w4_unet_c1, w_bb, wp4_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw4_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw4_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w4_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w4_unet_c1) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c3 * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_15 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_15 * hp4_unet_c1 * wp4_unet_c1 + (row_unet_c1 - 1) * wp4_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c3 + __icse_15) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + t_bpad[xi_unet_c1] * w_bb[wi_unet_c1]
    end
    bott[idx_unet_c1] = s_unet_c1 + b_bb[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_26!(__jacc_i, bott, bott_stack, n_b_out_unet_c1, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    __idx_bott_stack_0 = (i_unet_c1 - 1) + 1
    __cse_16 = bott[i_unet_c1]
    bott_stack[__idx_bott_stack_0] = __cse_16
    bott[i_unet_c1] = max(__cse_16, zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_b_27!(__jacc_i, bott, c3, hw2_unet_c1, hw4_unet_c1, scale_unet_c1, u2, w2_unet_c1, w4_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    oi_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    oj_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    oim1_unet_c1 = oi_unet_c1 - 1
    ojm1_unet_c1 = oj_unet_c1 - 1
    i_unet_c1 = div(oim1_unet_c1, scale_unet_c1) + 1
    j_unet_c1 = div(ojm1_unet_c1, scale_unet_c1) + 1
    xi_unet_c1 = (ci_unet_c1 - 1) * hw4_unet_c1 + (i_unet_c1 - 1) * w4_unet_c1 + j_unet_c1
    u2[idx_unet_c1] = bott[xi_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_28!(__jacc_i, c3, cat2, hw2_unet_c1, u2)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    cat2[idx_unet_c1] = u2[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_29!(__jacc_i, c2, c3, cat2, hw2_unet_c1, skip2)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    cat2[c3 * hw2_unet_c1 + idx_unet_c1] = skip2[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_30!(__jacc_i, cat2pad, n_cat2pad_unet_c1, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    cat2pad[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_b_31!(__jacc_i, c32_unet_c1, cat2, cat2pad, hp2_unet_c1, hw2_unet_c1, pad_unet_c1, w2_unet_c1, wp2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp2_unet_c1 * wp2_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp2_unet_c1 + (j_unet_c1 + pad_unet_c1)
    cat2pad[yi_unet_c1] = cat2[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_32!(__jacc_i, b_d2a, c2, c32_unet_c1, cat2pad, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d2, w2_unet_c1, w_d2a, wp2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c32_unet_c1 * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_17 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_17 * hp2_unet_c1 * wp2_unet_c1 + (row_unet_c1 - 1) * wp2_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c32_unet_c1 + __icse_17) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + cat2pad[xi_unet_c1] * w_d2a[wi_unet_c1]
    end
    t_d2[idx_unet_c1] = s_unet_c1 + b_d2a[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_33!(__jacc_i, n_d2_mid_unet_c1, t_d2, t_d2_stack, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    __idx_t_d2_stack_0 = (i_unet_c1 - 1) + 1
    __cse_18 = t_d2[i_unet_c1]
    t_d2_stack[__idx_t_d2_stack_0] = __cse_18
    t_d2[i_unet_c1] = max(__cse_18, zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_b_34!(__jacc_i, n_d2_midpad_unet_c1, t_d2pad, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    t_d2pad[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_b_35!(__jacc_i, c2, hp2_unet_c1, hw2_unet_c1, pad_unet_c1, t_d2, t_d2pad, w2_unet_c1, wp2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp2_unet_c1 * wp2_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp2_unet_c1 + (j_unet_c1 + pad_unet_c1)
    t_d2pad[yi_unet_c1] = t_d2[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_36!(__jacc_i, b_d2b, c2, dec2out, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d2pad, w2_unet_c1, w_d2b, wp2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c2 * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_19 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_19 * hp2_unet_c1 * wp2_unet_c1 + (row_unet_c1 - 1) * wp2_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c2 + __icse_19) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + t_d2pad[xi_unet_c1] * w_d2b[wi_unet_c1]
    end
    dec2out[idx_unet_c1] = s_unet_c1 + b_d2b[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_37!(__jacc_i, dec2out, dec2out_stack, n_d2_out_unet_c1, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    __idx_dec2out_stack_0 = (i_unet_c1 - 1) + 1
    __cse_20 = dec2out[i_unet_c1]
    dec2out_stack[__idx_dec2out_stack_0] = __cse_20
    dec2out[i_unet_c1] = max(__cse_20, zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_b_38!(__jacc_i, c2, dec2out, hw2_unet_c1, hw_unet_c1, scale_unet_c1, u1, w, w2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    oi_unet_c1 = div(rem_unet_c1, w) + 1
    oj_unet_c1 = mod(rem_unet_c1, w) + 1
    oim1_unet_c1 = oi_unet_c1 - 1
    ojm1_unet_c1 = oj_unet_c1 - 1
    i_unet_c1 = div(oim1_unet_c1, scale_unet_c1) + 1
    j_unet_c1 = div(ojm1_unet_c1, scale_unet_c1) + 1
    xi_unet_c1 = (ci_unet_c1 - 1) * hw2_unet_c1 + (i_unet_c1 - 1) * w2_unet_c1 + j_unet_c1
    u1[idx_unet_c1] = dec2out[xi_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_39!(__jacc_i, c2, cat1, hw_unet_c1, u1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    cat1[idx_unet_c1] = u1[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_40!(__jacc_i, c1, c2, cat1, hw_unet_c1, skip1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    cat1[c2 * hw_unet_c1 + idx_unet_c1] = skip1[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_41!(__jacc_i, cat1pad, n_cat1pad_unet_c1, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    cat1pad[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_b_42!(__jacc_i, c21_unet_c1, cat1, cat1pad, hp1_unet_c1, hw_unet_c1, pad_unet_c1, w, wp1_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp1_unet_c1 * wp1_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp1_unet_c1 + (j_unet_c1 + pad_unet_c1)
    cat1pad[yi_unet_c1] = cat1[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_43!(__jacc_i, b_d1a, c1, c21_unet_c1, cat1pad, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d1, w, w_d1a, wp1_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c21_unet_c1 * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_21 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_21 * hp1_unet_c1 * wp1_unet_c1 + (row_unet_c1 - 1) * wp1_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c21_unet_c1 + __icse_21) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + cat1pad[xi_unet_c1] * w_d1a[wi_unet_c1]
    end
    t_d1[idx_unet_c1] = s_unet_c1 + b_d1a[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_44!(__jacc_i, n_d1_mid_unet_c1, t_d1, t_d1_stack, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    __idx_t_d1_stack_0 = (i_unet_c1 - 1) + 1
    __cse_22 = t_d1[i_unet_c1]
    t_d1_stack[__idx_t_d1_stack_0] = __cse_22
    t_d1[i_unet_c1] = max(__cse_22, zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_b_45!(__jacc_i, n_d1_midpad_unet_c1, t_d1pad, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    t_d1pad[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_b_46!(__jacc_i, c1, hp1_unet_c1, hw_unet_c1, pad_unet_c1, t_d1, t_d1pad, w, wp1_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp1_unet_c1 * wp1_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp1_unet_c1 + (j_unet_c1 + pad_unet_c1)
    t_d1pad[yi_unet_c1] = t_d1[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_47!(__jacc_i, b_d1b, c1, dec1out, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d1pad, w, w_d1b, wp1_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c1 * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_23 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_23 * hp1_unet_c1 * wp1_unet_c1 + (row_unet_c1 - 1) * wp1_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c1 + __icse_23) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + t_d1pad[xi_unet_c1] * w_d1b[wi_unet_c1]
    end
    dec1out[idx_unet_c1] = s_unet_c1 + b_d1b[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_48!(__jacc_i, dec1out, dec1out_stack, n_d1_out_unet_c1, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    __idx_dec1out_stack_0 = (i_unet_c1 - 1) + 1
    __cse_24 = dec1out[i_unet_c1]
    dec1out_stack[__idx_dec1out_stack_0] = __cse_24
    dec1out[i_unet_c1] = max(__cse_24, zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_b_49!(__jacc_i, b_out, c1, c_out, dec1out, hw_unet_c1, kh_out_unet_c1, khkw_out_unet_c1, kw_out_unet_c1, w, w_out, y)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c1 * khkw_out_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_out_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_out_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_out_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_out_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_25 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_25 * hw_unet_c1 + (row_unet_c1 - 1) * w + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c1 + __icse_25) * kh_out_unet_c1 + (ki_unet_c1 - 1)) * kw_out_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + dec1out[xi_unet_c1] * w_out[wi_unet_c1]
    end
    y[idx_unet_c1] = s_unet_c1 + b_out[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_b_50!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_unet_loss_b_51!(__jacc_i, c_out, h, loss, target, w, y)
    i_o = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += (y[i_o] - target[i_o]) ^ 2
    return nothing
end

function jacc_kernel_unet_loss_b_52!(__jacc_i, c_out, h, lossb, target, targetb, w, y, yb)
    i_o = c_out * h * w + (__jacc_i - 1) * -1
    __cse_31 = (2 * (y[i_o] - target[i_o])) * lossb[1]
    yb[i_o] = yb[i_o] + __cse_31
    targetb[i_o] = targetb[i_o] + -__cse_31
    return nothing
end

function jacc_kernel_unet_loss_b_53!(__jacc_i, b_outb, c1, c_out, dec1out, dec1outb, hw_unet_c1, kh_out_unet_c1, khkw_out_unet_c1, kw_out_unet_c1, w, w_out, w_outb, yb)
    idx_unet_c1 = c_out * hw_unet_c1 + (__jacc_i - 1) * -1
    s_unet_c1b = 0.0
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    __oldb_0 = yb[idx_unet_c1]
    yb[idx_unet_c1] = 0.0
    s_unet_c1b = s_unet_c1b + __oldb_0
    Atomix.@atomic b_outb[co_unet_c1] += __oldb_0
    for i_k_unet_c1 = c1 * khkw_out_unet_c1:-1:1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_out_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_out_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_out_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_out_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_32 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_32 * hw_unet_c1 + (row_unet_c1 - 1) * w + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c1 + __icse_32) * kh_out_unet_c1 + (ki_unet_c1 - 1)) * kw_out_unet_c1 + kj_unet_c1
        Atomix.@atomic dec1outb[xi_unet_c1] += w_out[wi_unet_c1] * s_unet_c1b
        Atomix.@atomic w_outb[wi_unet_c1] += dec1out[xi_unet_c1] * s_unet_c1b
    end
    s_unet_c1b = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_54!(__jacc_i, dec1out, dec1out_stack, dec1outb, n_d1_out_unet_c1, zero_val_unet_c1)
    i_unet_c1 = n_d1_out_unet_c1 + (__jacc_i - 1) * -1
    __idx_dec1out_stack_0 = (i_unet_c1 - 1) + 1
    dec1out[i_unet_c1] = dec1out_stack[__idx_dec1out_stack_0]
    __oldb_2 = dec1outb[i_unet_c1]
    dec1outb[i_unet_c1] = 0.0
    dec1outb[i_unet_c1] = dec1outb[i_unet_c1] + (0.5 * (1.0 + sign(dec1out[i_unet_c1] - zero_val_unet_c1))) * __oldb_2
    return nothing
end

function jacc_kernel_unet_loss_b_55!(__jacc_i, b_d1bb, c1, dec1outb, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d1pad, t_d1padb, w, w_d1b, w_d1bb, wp1_unet_c1)
    idx_unet_c1 = c1 * hw_unet_c1 + (__jacc_i - 1) * -1
    s_unet_c1b = 0.0
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    __oldb_0 = dec1outb[idx_unet_c1]
    dec1outb[idx_unet_c1] = 0.0
    s_unet_c1b = s_unet_c1b + __oldb_0
    Atomix.@atomic b_d1bb[co_unet_c1] += __oldb_0
    for i_k_unet_c1 = c1 * khkw_unet_c1:-1:1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_33 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_33 * hp1_unet_c1 * wp1_unet_c1 + (row_unet_c1 - 1) * wp1_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c1 + __icse_33) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        Atomix.@atomic t_d1padb[xi_unet_c1] += w_d1b[wi_unet_c1] * s_unet_c1b
        Atomix.@atomic w_d1bb[wi_unet_c1] += t_d1pad[xi_unet_c1] * s_unet_c1b
    end
    s_unet_c1b = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_56!(__jacc_i, c1, hp1_unet_c1, hw_unet_c1, pad_unet_c1, t_d1b, t_d1padb, w, wp1_unet_c1)
    idx_unet_c1 = c1 * hw_unet_c1 + (__jacc_i - 1) * -1
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp1_unet_c1 * wp1_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp1_unet_c1 + (j_unet_c1 + pad_unet_c1)
    __oldb_0 = t_d1padb[yi_unet_c1]
    t_d1padb[yi_unet_c1] = 0.0
    t_d1b[idx_unet_c1] = t_d1b[idx_unet_c1] + __oldb_0
    return nothing
end

function jacc_kernel_unet_loss_b_57!(__jacc_i, n_d1_midpad_unet_c1, t_d1padb)
    i_unet_c1 = n_d1_midpad_unet_c1 + (__jacc_i - 1) * -1
    t_d1padb[i_unet_c1] = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_58!(__jacc_i, n_d1_mid_unet_c1, t_d1, t_d1_stack, t_d1b, zero_val_unet_c1)
    i_unet_c1 = n_d1_mid_unet_c1 + (__jacc_i - 1) * -1
    __idx_t_d1_stack_0 = (i_unet_c1 - 1) + 1
    t_d1[i_unet_c1] = t_d1_stack[__idx_t_d1_stack_0]
    __oldb_2 = t_d1b[i_unet_c1]
    t_d1b[i_unet_c1] = 0.0
    t_d1b[i_unet_c1] = t_d1b[i_unet_c1] + (0.5 * (1.0 + sign(t_d1[i_unet_c1] - zero_val_unet_c1))) * __oldb_2
    return nothing
end

function jacc_kernel_unet_loss_b_59!(__jacc_i, b_d1ab, c1, c21_unet_c1, cat1pad, cat1padb, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d1b, w, w_d1a, w_d1ab, wp1_unet_c1)
    idx_unet_c1 = c1 * hw_unet_c1 + (__jacc_i - 1) * -1
    s_unet_c1b = 0.0
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    __oldb_0 = t_d1b[idx_unet_c1]
    t_d1b[idx_unet_c1] = 0.0
    s_unet_c1b = s_unet_c1b + __oldb_0
    Atomix.@atomic b_d1ab[co_unet_c1] += __oldb_0
    for i_k_unet_c1 = c21_unet_c1 * khkw_unet_c1:-1:1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_34 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_34 * hp1_unet_c1 * wp1_unet_c1 + (row_unet_c1 - 1) * wp1_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c21_unet_c1 + __icse_34) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        Atomix.@atomic cat1padb[xi_unet_c1] += w_d1a[wi_unet_c1] * s_unet_c1b
        Atomix.@atomic w_d1ab[wi_unet_c1] += cat1pad[xi_unet_c1] * s_unet_c1b
    end
    s_unet_c1b = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_60!(__jacc_i, c21_unet_c1, cat1b, cat1padb, hp1_unet_c1, hw_unet_c1, pad_unet_c1, w, wp1_unet_c1)
    idx_unet_c1 = c21_unet_c1 * hw_unet_c1 + (__jacc_i - 1) * -1
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp1_unet_c1 * wp1_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp1_unet_c1 + (j_unet_c1 + pad_unet_c1)
    __oldb_0 = cat1padb[yi_unet_c1]
    cat1padb[yi_unet_c1] = 0.0
    cat1b[idx_unet_c1] = cat1b[idx_unet_c1] + __oldb_0
    return nothing
end

function jacc_kernel_unet_loss_b_61!(__jacc_i, cat1padb, n_cat1pad_unet_c1)
    i_unet_c1 = n_cat1pad_unet_c1 + (__jacc_i - 1) * -1
    cat1padb[i_unet_c1] = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_62!(__jacc_i, c1, c2, cat1b, hw_unet_c1, skip1b)
    idx_unet_c1 = c1 * hw_unet_c1 + (__jacc_i - 1) * -1
    __oldb_0 = cat1b[c2 * hw_unet_c1 + idx_unet_c1]
    cat1b[c2 * hw_unet_c1 + idx_unet_c1] = 0.0
    skip1b[idx_unet_c1] = skip1b[idx_unet_c1] + __oldb_0
    return nothing
end

function jacc_kernel_unet_loss_b_63!(__jacc_i, c2, cat1b, hw_unet_c1, u1b)
    idx_unet_c1 = c2 * hw_unet_c1 + (__jacc_i - 1) * -1
    __oldb_0 = cat1b[idx_unet_c1]
    cat1b[idx_unet_c1] = 0.0
    u1b[idx_unet_c1] = u1b[idx_unet_c1] + __oldb_0
    return nothing
end

function jacc_kernel_unet_loss_b_64!(__jacc_i, c2, dec2outb, hw2_unet_c1, hw_unet_c1, scale_unet_c1, u1b, w, w2_unet_c1)
    idx_unet_c1 = c2 * hw_unet_c1 + (__jacc_i - 1) * -1
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    oi_unet_c1 = div(rem_unet_c1, w) + 1
    oj_unet_c1 = mod(rem_unet_c1, w) + 1
    oim1_unet_c1 = oi_unet_c1 - 1
    ojm1_unet_c1 = oj_unet_c1 - 1
    i_unet_c1 = div(oim1_unet_c1, scale_unet_c1) + 1
    j_unet_c1 = div(ojm1_unet_c1, scale_unet_c1) + 1
    xi_unet_c1 = (ci_unet_c1 - 1) * hw2_unet_c1 + (i_unet_c1 - 1) * w2_unet_c1 + j_unet_c1
    __oldb_0 = u1b[idx_unet_c1]
    u1b[idx_unet_c1] = 0.0
    Atomix.@atomic dec2outb[xi_unet_c1] += __oldb_0
    return nothing
end

function jacc_kernel_unet_loss_b_65!(__jacc_i, dec2out, dec2out_stack, dec2outb, n_d2_out_unet_c1, zero_val_unet_c1)
    i_unet_c1 = n_d2_out_unet_c1 + (__jacc_i - 1) * -1
    __idx_dec2out_stack_0 = (i_unet_c1 - 1) + 1
    dec2out[i_unet_c1] = dec2out_stack[__idx_dec2out_stack_0]
    __oldb_2 = dec2outb[i_unet_c1]
    dec2outb[i_unet_c1] = 0.0
    dec2outb[i_unet_c1] = dec2outb[i_unet_c1] + (0.5 * (1.0 + sign(dec2out[i_unet_c1] - zero_val_unet_c1))) * __oldb_2
    return nothing
end

function jacc_kernel_unet_loss_b_66!(__jacc_i, b_d2bb, c2, dec2outb, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d2pad, t_d2padb, w2_unet_c1, w_d2b, w_d2bb, wp2_unet_c1)
    idx_unet_c1 = c2 * hw2_unet_c1 + (__jacc_i - 1) * -1
    s_unet_c1b = 0.0
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    __oldb_0 = dec2outb[idx_unet_c1]
    dec2outb[idx_unet_c1] = 0.0
    s_unet_c1b = s_unet_c1b + __oldb_0
    Atomix.@atomic b_d2bb[co_unet_c1] += __oldb_0
    for i_k_unet_c1 = c2 * khkw_unet_c1:-1:1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_35 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_35 * hp2_unet_c1 * wp2_unet_c1 + (row_unet_c1 - 1) * wp2_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c2 + __icse_35) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        Atomix.@atomic t_d2padb[xi_unet_c1] += w_d2b[wi_unet_c1] * s_unet_c1b
        Atomix.@atomic w_d2bb[wi_unet_c1] += t_d2pad[xi_unet_c1] * s_unet_c1b
    end
    s_unet_c1b = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_67!(__jacc_i, c2, hp2_unet_c1, hw2_unet_c1, pad_unet_c1, t_d2b, t_d2padb, w2_unet_c1, wp2_unet_c1)
    idx_unet_c1 = c2 * hw2_unet_c1 + (__jacc_i - 1) * -1
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp2_unet_c1 * wp2_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp2_unet_c1 + (j_unet_c1 + pad_unet_c1)
    __oldb_0 = t_d2padb[yi_unet_c1]
    t_d2padb[yi_unet_c1] = 0.0
    t_d2b[idx_unet_c1] = t_d2b[idx_unet_c1] + __oldb_0
    return nothing
end

function jacc_kernel_unet_loss_b_68!(__jacc_i, n_d2_midpad_unet_c1, t_d2padb)
    i_unet_c1 = n_d2_midpad_unet_c1 + (__jacc_i - 1) * -1
    t_d2padb[i_unet_c1] = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_69!(__jacc_i, n_d2_mid_unet_c1, t_d2, t_d2_stack, t_d2b, zero_val_unet_c1)
    i_unet_c1 = n_d2_mid_unet_c1 + (__jacc_i - 1) * -1
    __idx_t_d2_stack_0 = (i_unet_c1 - 1) + 1
    t_d2[i_unet_c1] = t_d2_stack[__idx_t_d2_stack_0]
    __oldb_2 = t_d2b[i_unet_c1]
    t_d2b[i_unet_c1] = 0.0
    t_d2b[i_unet_c1] = t_d2b[i_unet_c1] + (0.5 * (1.0 + sign(t_d2[i_unet_c1] - zero_val_unet_c1))) * __oldb_2
    return nothing
end

function jacc_kernel_unet_loss_b_70!(__jacc_i, b_d2ab, c2, c32_unet_c1, cat2pad, cat2padb, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d2b, w2_unet_c1, w_d2a, w_d2ab, wp2_unet_c1)
    idx_unet_c1 = c2 * hw2_unet_c1 + (__jacc_i - 1) * -1
    s_unet_c1b = 0.0
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    __oldb_0 = t_d2b[idx_unet_c1]
    t_d2b[idx_unet_c1] = 0.0
    s_unet_c1b = s_unet_c1b + __oldb_0
    Atomix.@atomic b_d2ab[co_unet_c1] += __oldb_0
    for i_k_unet_c1 = c32_unet_c1 * khkw_unet_c1:-1:1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_36 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_36 * hp2_unet_c1 * wp2_unet_c1 + (row_unet_c1 - 1) * wp2_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c32_unet_c1 + __icse_36) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        Atomix.@atomic cat2padb[xi_unet_c1] += w_d2a[wi_unet_c1] * s_unet_c1b
        Atomix.@atomic w_d2ab[wi_unet_c1] += cat2pad[xi_unet_c1] * s_unet_c1b
    end
    s_unet_c1b = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_71!(__jacc_i, c32_unet_c1, cat2b, cat2padb, hp2_unet_c1, hw2_unet_c1, pad_unet_c1, w2_unet_c1, wp2_unet_c1)
    idx_unet_c1 = c32_unet_c1 * hw2_unet_c1 + (__jacc_i - 1) * -1
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp2_unet_c1 * wp2_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp2_unet_c1 + (j_unet_c1 + pad_unet_c1)
    __oldb_0 = cat2padb[yi_unet_c1]
    cat2padb[yi_unet_c1] = 0.0
    cat2b[idx_unet_c1] = cat2b[idx_unet_c1] + __oldb_0
    return nothing
end

function jacc_kernel_unet_loss_b_72!(__jacc_i, cat2padb, n_cat2pad_unet_c1)
    i_unet_c1 = n_cat2pad_unet_c1 + (__jacc_i - 1) * -1
    cat2padb[i_unet_c1] = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_73!(__jacc_i, c2, c3, cat2b, hw2_unet_c1, skip2b)
    idx_unet_c1 = c2 * hw2_unet_c1 + (__jacc_i - 1) * -1
    __oldb_0 = cat2b[c3 * hw2_unet_c1 + idx_unet_c1]
    cat2b[c3 * hw2_unet_c1 + idx_unet_c1] = 0.0
    skip2b[idx_unet_c1] = skip2b[idx_unet_c1] + __oldb_0
    return nothing
end

function jacc_kernel_unet_loss_b_74!(__jacc_i, c3, cat2b, hw2_unet_c1, u2b)
    idx_unet_c1 = c3 * hw2_unet_c1 + (__jacc_i - 1) * -1
    __oldb_0 = cat2b[idx_unet_c1]
    cat2b[idx_unet_c1] = 0.0
    u2b[idx_unet_c1] = u2b[idx_unet_c1] + __oldb_0
    return nothing
end

function jacc_kernel_unet_loss_b_75!(__jacc_i, bottb, c3, hw2_unet_c1, hw4_unet_c1, scale_unet_c1, u2b, w2_unet_c1, w4_unet_c1)
    idx_unet_c1 = c3 * hw2_unet_c1 + (__jacc_i - 1) * -1
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    oi_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    oj_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    oim1_unet_c1 = oi_unet_c1 - 1
    ojm1_unet_c1 = oj_unet_c1 - 1
    i_unet_c1 = div(oim1_unet_c1, scale_unet_c1) + 1
    j_unet_c1 = div(ojm1_unet_c1, scale_unet_c1) + 1
    xi_unet_c1 = (ci_unet_c1 - 1) * hw4_unet_c1 + (i_unet_c1 - 1) * w4_unet_c1 + j_unet_c1
    __oldb_0 = u2b[idx_unet_c1]
    u2b[idx_unet_c1] = 0.0
    Atomix.@atomic bottb[xi_unet_c1] += __oldb_0
    return nothing
end

function jacc_kernel_unet_loss_b_76!(__jacc_i, bott, bott_stack, bottb, n_b_out_unet_c1, zero_val_unet_c1)
    i_unet_c1 = n_b_out_unet_c1 + (__jacc_i - 1) * -1
    __idx_bott_stack_0 = (i_unet_c1 - 1) + 1
    bott[i_unet_c1] = bott_stack[__idx_bott_stack_0]
    __oldb_2 = bottb[i_unet_c1]
    bottb[i_unet_c1] = 0.0
    bottb[i_unet_c1] = bottb[i_unet_c1] + (0.5 * (1.0 + sign(bott[i_unet_c1] - zero_val_unet_c1))) * __oldb_2
    return nothing
end

function jacc_kernel_unet_loss_b_77!(__jacc_i, b_bbb, bottb, c3, hp4_unet_c1, hw4_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_bpad, t_bpadb, w4_unet_c1, w_bb, w_bbb, wp4_unet_c1)
    idx_unet_c1 = c3 * hw4_unet_c1 + (__jacc_i - 1) * -1
    s_unet_c1b = 0.0
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw4_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw4_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w4_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w4_unet_c1) + 1
    __oldb_0 = bottb[idx_unet_c1]
    bottb[idx_unet_c1] = 0.0
    s_unet_c1b = s_unet_c1b + __oldb_0
    Atomix.@atomic b_bbb[co_unet_c1] += __oldb_0
    for i_k_unet_c1 = c3 * khkw_unet_c1:-1:1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_37 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_37 * hp4_unet_c1 * wp4_unet_c1 + (row_unet_c1 - 1) * wp4_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c3 + __icse_37) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        Atomix.@atomic t_bpadb[xi_unet_c1] += w_bb[wi_unet_c1] * s_unet_c1b
        Atomix.@atomic w_bbb[wi_unet_c1] += t_bpad[xi_unet_c1] * s_unet_c1b
    end
    s_unet_c1b = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_78!(__jacc_i, c3, hp4_unet_c1, hw4_unet_c1, pad_unet_c1, t_bb, t_bpadb, w4_unet_c1, wp4_unet_c1)
    idx_unet_c1 = c3 * hw4_unet_c1 + (__jacc_i - 1) * -1
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw4_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw4_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w4_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w4_unet_c1) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp4_unet_c1 * wp4_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp4_unet_c1 + (j_unet_c1 + pad_unet_c1)
    __oldb_0 = t_bpadb[yi_unet_c1]
    t_bpadb[yi_unet_c1] = 0.0
    t_bb[idx_unet_c1] = t_bb[idx_unet_c1] + __oldb_0
    return nothing
end

function jacc_kernel_unet_loss_b_79!(__jacc_i, n_b_midpad_unet_c1, t_bpadb)
    i_unet_c1 = n_b_midpad_unet_c1 + (__jacc_i - 1) * -1
    t_bpadb[i_unet_c1] = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_80!(__jacc_i, n_b_mid_unet_c1, t_b, t_b_stack, t_bb, zero_val_unet_c1)
    i_unet_c1 = n_b_mid_unet_c1 + (__jacc_i - 1) * -1
    __idx_t_b_stack_0 = (i_unet_c1 - 1) + 1
    t_b[i_unet_c1] = t_b_stack[__idx_t_b_stack_0]
    __oldb_2 = t_bb[i_unet_c1]
    t_bb[i_unet_c1] = 0.0
    t_bb[i_unet_c1] = t_bb[i_unet_c1] + (0.5 * (1.0 + sign(t_b[i_unet_c1] - zero_val_unet_c1))) * __oldb_2
    return nothing
end

function jacc_kernel_unet_loss_b_81!(__jacc_i, b_bab, c2, c3, hp4_unet_c1, hw4_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, p2pad, p2padb, t_bb, w4_unet_c1, w_ba, w_bab, wp4_unet_c1)
    idx_unet_c1 = c3 * hw4_unet_c1 + (__jacc_i - 1) * -1
    s_unet_c1b = 0.0
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw4_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw4_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w4_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w4_unet_c1) + 1
    __oldb_0 = t_bb[idx_unet_c1]
    t_bb[idx_unet_c1] = 0.0
    s_unet_c1b = s_unet_c1b + __oldb_0
    Atomix.@atomic b_bab[co_unet_c1] += __oldb_0
    for i_k_unet_c1 = c2 * khkw_unet_c1:-1:1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_38 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_38 * hp4_unet_c1 * wp4_unet_c1 + (row_unet_c1 - 1) * wp4_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c2 + __icse_38) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        Atomix.@atomic p2padb[xi_unet_c1] += w_ba[wi_unet_c1] * s_unet_c1b
        Atomix.@atomic w_bab[wi_unet_c1] += p2pad[xi_unet_c1] * s_unet_c1b
    end
    s_unet_c1b = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_82!(__jacc_i, c2, hp4_unet_c1, hw4_unet_c1, p2b, p2padb, pad_unet_c1, w4_unet_c1, wp4_unet_c1)
    idx_unet_c1 = c2 * hw4_unet_c1 + (__jacc_i - 1) * -1
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw4_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw4_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w4_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w4_unet_c1) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp4_unet_c1 * wp4_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp4_unet_c1 + (j_unet_c1 + pad_unet_c1)
    __oldb_0 = p2padb[yi_unet_c1]
    p2padb[yi_unet_c1] = 0.0
    p2b[idx_unet_c1] = p2b[idx_unet_c1] + __oldb_0
    return nothing
end

function jacc_kernel_unet_loss_b_83!(__jacc_i, n_p2pad_unet_c1, p2padb)
    i_unet_c1 = n_p2pad_unet_c1 + (__jacc_i - 1) * -1
    p2padb[i_unet_c1] = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_84!(__jacc_i, c2, hw2_unet_c1, hw4_unet_c1, p2b, skip2, skip2b, w2_unet_c1, w4_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    a12_unet_c1b = 0.0
    a22_unet_c1b = 0.0
    m2_unet_c1b = 0.0
    m1_unet_c1b = 0.0
    a21_unet_c1b = 0.0
    a11_unet_c1b = 0.0
    __icse_39 = idx_unet_c1 - 1
    idxm1_unet_c1 = __icse_39
    ci_unet_c1 = div(idxm1_unet_c1, hw4_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw4_unet_c1)
    oi_unet_c1 = div(rem_unet_c1, w4_unet_c1) + 1
    oj_unet_c1 = mod(rem_unet_c1, w4_unet_c1) + 1
    i0_unet_c1 = 2oi_unet_c1 - 1
    j0_unet_c1 = 2oj_unet_c1 - 1
    a11_unet_c1 = skip2[(ci_unet_c1 - 1) * hw2_unet_c1 + (i0_unet_c1 - 1) * w2_unet_c1 + j0_unet_c1]
    a12_unet_c1 = skip2[(ci_unet_c1 - 1) * hw2_unet_c1 + (i0_unet_c1 - 1) * w2_unet_c1 + j0_unet_c1 + 1]
    a21_unet_c1 = skip2[(ci_unet_c1 - 1) * hw2_unet_c1 + i0_unet_c1 * w2_unet_c1 + j0_unet_c1]
    a22_unet_c1 = skip2[(ci_unet_c1 - 1) * hw2_unet_c1 + i0_unet_c1 * w2_unet_c1 + j0_unet_c1 + 1]
    m1_unet_c1 = max(a11_unet_c1, a12_unet_c1)
    m2_unet_c1 = max(a21_unet_c1, a22_unet_c1)
    idxm1_unet_c1 = __icse_39
    ci_unet_c1 = div(idxm1_unet_c1, hw4_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw4_unet_c1)
    oi_unet_c1 = div(rem_unet_c1, w4_unet_c1) + 1
    oj_unet_c1 = mod(rem_unet_c1, w4_unet_c1) + 1
    i0_unet_c1 = 2oi_unet_c1 - 1
    j0_unet_c1 = 2oj_unet_c1 - 1
    __oldb_0 = p2b[idx_unet_c1]
    p2b[idx_unet_c1] = 0.0
    m1_unet_c1b = m1_unet_c1b + (0.5 * (1.0 + sign(m1_unet_c1 - m2_unet_c1))) * __oldb_0
    m2_unet_c1b = m2_unet_c1b + (0.5 * (1.0 + sign(m2_unet_c1 - m1_unet_c1))) * __oldb_0
    __oldb_0 = m2_unet_c1b
    m2_unet_c1b = 0.0
    a21_unet_c1b = a21_unet_c1b + (0.5 * (1.0 + sign(a21_unet_c1 - a22_unet_c1))) * __oldb_0
    a22_unet_c1b = a22_unet_c1b + (0.5 * (1.0 + sign(a22_unet_c1 - a21_unet_c1))) * __oldb_0
    __oldb_0 = m1_unet_c1b
    m1_unet_c1b = 0.0
    a11_unet_c1b = a11_unet_c1b + (0.5 * (1.0 + sign(a11_unet_c1 - a12_unet_c1))) * __oldb_0
    a12_unet_c1b = a12_unet_c1b + (0.5 * (1.0 + sign(a12_unet_c1 - a11_unet_c1))) * __oldb_0
    __oldb_0 = a22_unet_c1b
    a22_unet_c1b = 0.0
    Atomix.@atomic skip2b[(ci_unet_c1 - 1) * hw2_unet_c1 + i0_unet_c1 * w2_unet_c1 + j0_unet_c1 + 1] += __oldb_0
    __oldb_0 = a21_unet_c1b
    a21_unet_c1b = 0.0
    Atomix.@atomic skip2b[(ci_unet_c1 - 1) * hw2_unet_c1 + i0_unet_c1 * w2_unet_c1 + j0_unet_c1] += __oldb_0
    __oldb_0 = a12_unet_c1b
    a12_unet_c1b = 0.0
    Atomix.@atomic skip2b[(ci_unet_c1 - 1) * hw2_unet_c1 + (i0_unet_c1 - 1) * w2_unet_c1 + j0_unet_c1 + 1] += __oldb_0
    __oldb_0 = a11_unet_c1b
    a11_unet_c1b = 0.0
    Atomix.@atomic skip2b[(ci_unet_c1 - 1) * hw2_unet_c1 + (i0_unet_c1 - 1) * w2_unet_c1 + j0_unet_c1] += __oldb_0
    return nothing
end

function jacc_kernel_unet_loss_b_85!(__jacc_i, n_e2_out_unet_c1, skip2, skip2_stack, skip2b, zero_val_unet_c1)
    i_unet_c1 = n_e2_out_unet_c1 + (__jacc_i - 1) * -1
    __idx_skip2_stack_0 = (i_unet_c1 - 1) + 1
    skip2[i_unet_c1] = skip2_stack[__idx_skip2_stack_0]
    __oldb_2 = skip2b[i_unet_c1]
    skip2b[i_unet_c1] = 0.0
    skip2b[i_unet_c1] = skip2b[i_unet_c1] + (0.5 * (1.0 + sign(skip2[i_unet_c1] - zero_val_unet_c1))) * __oldb_2
    return nothing
end

function jacc_kernel_unet_loss_b_86!(__jacc_i, b_e2bb, c2, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, skip2b, t_e2pad, t_e2padb, w2_unet_c1, w_e2b, w_e2bb, wp2_unet_c1)
    idx_unet_c1 = c2 * hw2_unet_c1 + (__jacc_i - 1) * -1
    s_unet_c1b = 0.0
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    __oldb_0 = skip2b[idx_unet_c1]
    skip2b[idx_unet_c1] = 0.0
    s_unet_c1b = s_unet_c1b + __oldb_0
    Atomix.@atomic b_e2bb[co_unet_c1] += __oldb_0
    for i_k_unet_c1 = c2 * khkw_unet_c1:-1:1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_40 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_40 * hp2_unet_c1 * wp2_unet_c1 + (row_unet_c1 - 1) * wp2_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c2 + __icse_40) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        Atomix.@atomic t_e2padb[xi_unet_c1] += w_e2b[wi_unet_c1] * s_unet_c1b
        Atomix.@atomic w_e2bb[wi_unet_c1] += t_e2pad[xi_unet_c1] * s_unet_c1b
    end
    s_unet_c1b = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_87!(__jacc_i, c2, hp2_unet_c1, hw2_unet_c1, pad_unet_c1, t_e2b, t_e2padb, w2_unet_c1, wp2_unet_c1)
    idx_unet_c1 = c2 * hw2_unet_c1 + (__jacc_i - 1) * -1
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp2_unet_c1 * wp2_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp2_unet_c1 + (j_unet_c1 + pad_unet_c1)
    __oldb_0 = t_e2padb[yi_unet_c1]
    t_e2padb[yi_unet_c1] = 0.0
    t_e2b[idx_unet_c1] = t_e2b[idx_unet_c1] + __oldb_0
    return nothing
end

function jacc_kernel_unet_loss_b_88!(__jacc_i, n_e2_midpad_unet_c1, t_e2padb)
    i_unet_c1 = n_e2_midpad_unet_c1 + (__jacc_i - 1) * -1
    t_e2padb[i_unet_c1] = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_89!(__jacc_i, n_e2_mid_unet_c1, t_e2, t_e2_stack, t_e2b, zero_val_unet_c1)
    i_unet_c1 = n_e2_mid_unet_c1 + (__jacc_i - 1) * -1
    __idx_t_e2_stack_0 = (i_unet_c1 - 1) + 1
    t_e2[i_unet_c1] = t_e2_stack[__idx_t_e2_stack_0]
    __oldb_2 = t_e2b[i_unet_c1]
    t_e2b[i_unet_c1] = 0.0
    t_e2b[i_unet_c1] = t_e2b[i_unet_c1] + (0.5 * (1.0 + sign(t_e2[i_unet_c1] - zero_val_unet_c1))) * __oldb_2
    return nothing
end

function jacc_kernel_unet_loss_b_90!(__jacc_i, b_e2ab, c1, c2, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, p1pad, p1padb, t_e2b, w2_unet_c1, w_e2a, w_e2ab, wp2_unet_c1)
    idx_unet_c1 = c2 * hw2_unet_c1 + (__jacc_i - 1) * -1
    s_unet_c1b = 0.0
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    __oldb_0 = t_e2b[idx_unet_c1]
    t_e2b[idx_unet_c1] = 0.0
    s_unet_c1b = s_unet_c1b + __oldb_0
    Atomix.@atomic b_e2ab[co_unet_c1] += __oldb_0
    for i_k_unet_c1 = c1 * khkw_unet_c1:-1:1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_41 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_41 * hp2_unet_c1 * wp2_unet_c1 + (row_unet_c1 - 1) * wp2_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c1 + __icse_41) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        Atomix.@atomic p1padb[xi_unet_c1] += w_e2a[wi_unet_c1] * s_unet_c1b
        Atomix.@atomic w_e2ab[wi_unet_c1] += p1pad[xi_unet_c1] * s_unet_c1b
    end
    s_unet_c1b = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_91!(__jacc_i, c1, hp2_unet_c1, hw2_unet_c1, p1b, p1padb, pad_unet_c1, w2_unet_c1, wp2_unet_c1)
    idx_unet_c1 = c1 * hw2_unet_c1 + (__jacc_i - 1) * -1
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp2_unet_c1 * wp2_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp2_unet_c1 + (j_unet_c1 + pad_unet_c1)
    __oldb_0 = p1padb[yi_unet_c1]
    p1padb[yi_unet_c1] = 0.0
    p1b[idx_unet_c1] = p1b[idx_unet_c1] + __oldb_0
    return nothing
end

function jacc_kernel_unet_loss_b_92!(__jacc_i, n_p1pad_unet_c1, p1padb)
    i_unet_c1 = n_p1pad_unet_c1 + (__jacc_i - 1) * -1
    p1padb[i_unet_c1] = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_93!(__jacc_i, c1, hw2_unet_c1, hw_unet_c1, p1b, skip1, skip1b, w, w2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    a12_unet_c1b = 0.0
    a22_unet_c1b = 0.0
    m2_unet_c1b = 0.0
    m1_unet_c1b = 0.0
    a21_unet_c1b = 0.0
    a11_unet_c1b = 0.0
    __icse_42 = idx_unet_c1 - 1
    idxm1_unet_c1 = __icse_42
    ci_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    oi_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    oj_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    i0_unet_c1 = 2oi_unet_c1 - 1
    j0_unet_c1 = 2oj_unet_c1 - 1
    a11_unet_c1 = skip1[(ci_unet_c1 - 1) * hw_unet_c1 + (i0_unet_c1 - 1) * w + j0_unet_c1]
    a12_unet_c1 = skip1[(ci_unet_c1 - 1) * hw_unet_c1 + (i0_unet_c1 - 1) * w + j0_unet_c1 + 1]
    a21_unet_c1 = skip1[(ci_unet_c1 - 1) * hw_unet_c1 + i0_unet_c1 * w + j0_unet_c1]
    a22_unet_c1 = skip1[(ci_unet_c1 - 1) * hw_unet_c1 + i0_unet_c1 * w + j0_unet_c1 + 1]
    m1_unet_c1 = max(a11_unet_c1, a12_unet_c1)
    m2_unet_c1 = max(a21_unet_c1, a22_unet_c1)
    idxm1_unet_c1 = __icse_42
    ci_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    oi_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    oj_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    i0_unet_c1 = 2oi_unet_c1 - 1
    j0_unet_c1 = 2oj_unet_c1 - 1
    __oldb_0 = p1b[idx_unet_c1]
    p1b[idx_unet_c1] = 0.0
    m1_unet_c1b = m1_unet_c1b + (0.5 * (1.0 + sign(m1_unet_c1 - m2_unet_c1))) * __oldb_0
    m2_unet_c1b = m2_unet_c1b + (0.5 * (1.0 + sign(m2_unet_c1 - m1_unet_c1))) * __oldb_0
    __oldb_0 = m2_unet_c1b
    m2_unet_c1b = 0.0
    a21_unet_c1b = a21_unet_c1b + (0.5 * (1.0 + sign(a21_unet_c1 - a22_unet_c1))) * __oldb_0
    a22_unet_c1b = a22_unet_c1b + (0.5 * (1.0 + sign(a22_unet_c1 - a21_unet_c1))) * __oldb_0
    __oldb_0 = m1_unet_c1b
    m1_unet_c1b = 0.0
    a11_unet_c1b = a11_unet_c1b + (0.5 * (1.0 + sign(a11_unet_c1 - a12_unet_c1))) * __oldb_0
    a12_unet_c1b = a12_unet_c1b + (0.5 * (1.0 + sign(a12_unet_c1 - a11_unet_c1))) * __oldb_0
    __oldb_0 = a22_unet_c1b
    a22_unet_c1b = 0.0
    Atomix.@atomic skip1b[(ci_unet_c1 - 1) * hw_unet_c1 + i0_unet_c1 * w + j0_unet_c1 + 1] += __oldb_0
    __oldb_0 = a21_unet_c1b
    a21_unet_c1b = 0.0
    Atomix.@atomic skip1b[(ci_unet_c1 - 1) * hw_unet_c1 + i0_unet_c1 * w + j0_unet_c1] += __oldb_0
    __oldb_0 = a12_unet_c1b
    a12_unet_c1b = 0.0
    Atomix.@atomic skip1b[(ci_unet_c1 - 1) * hw_unet_c1 + (i0_unet_c1 - 1) * w + j0_unet_c1 + 1] += __oldb_0
    __oldb_0 = a11_unet_c1b
    a11_unet_c1b = 0.0
    Atomix.@atomic skip1b[(ci_unet_c1 - 1) * hw_unet_c1 + (i0_unet_c1 - 1) * w + j0_unet_c1] += __oldb_0
    return nothing
end

function jacc_kernel_unet_loss_b_94!(__jacc_i, n_e1_out_unet_c1, skip1, skip1_stack, skip1b, zero_val_unet_c1)
    i_unet_c1 = n_e1_out_unet_c1 + (__jacc_i - 1) * -1
    __idx_skip1_stack_0 = (i_unet_c1 - 1) + 1
    skip1[i_unet_c1] = skip1_stack[__idx_skip1_stack_0]
    __oldb_2 = skip1b[i_unet_c1]
    skip1b[i_unet_c1] = 0.0
    skip1b[i_unet_c1] = skip1b[i_unet_c1] + (0.5 * (1.0 + sign(skip1[i_unet_c1] - zero_val_unet_c1))) * __oldb_2
    return nothing
end

function jacc_kernel_unet_loss_b_95!(__jacc_i, b_e1bb, c1, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, skip1b, t_e1pad, t_e1padb, w, w_e1b, w_e1bb, wp1_unet_c1)
    idx_unet_c1 = c1 * hw_unet_c1 + (__jacc_i - 1) * -1
    s_unet_c1b = 0.0
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    __oldb_0 = skip1b[idx_unet_c1]
    skip1b[idx_unet_c1] = 0.0
    s_unet_c1b = s_unet_c1b + __oldb_0
    Atomix.@atomic b_e1bb[co_unet_c1] += __oldb_0
    for i_k_unet_c1 = c1 * khkw_unet_c1:-1:1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_43 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_43 * hp1_unet_c1 * wp1_unet_c1 + (row_unet_c1 - 1) * wp1_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c1 + __icse_43) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        Atomix.@atomic t_e1padb[xi_unet_c1] += w_e1b[wi_unet_c1] * s_unet_c1b
        Atomix.@atomic w_e1bb[wi_unet_c1] += t_e1pad[xi_unet_c1] * s_unet_c1b
    end
    s_unet_c1b = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_96!(__jacc_i, c1, hp1_unet_c1, hw_unet_c1, pad_unet_c1, t_e1b, t_e1padb, w, wp1_unet_c1)
    idx_unet_c1 = c1 * hw_unet_c1 + (__jacc_i - 1) * -1
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp1_unet_c1 * wp1_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp1_unet_c1 + (j_unet_c1 + pad_unet_c1)
    __oldb_0 = t_e1padb[yi_unet_c1]
    t_e1padb[yi_unet_c1] = 0.0
    t_e1b[idx_unet_c1] = t_e1b[idx_unet_c1] + __oldb_0
    return nothing
end

function jacc_kernel_unet_loss_b_97!(__jacc_i, n_e1_midpad_unet_c1, t_e1padb)
    i_unet_c1 = n_e1_midpad_unet_c1 + (__jacc_i - 1) * -1
    t_e1padb[i_unet_c1] = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_98!(__jacc_i, n_e1_mid_unet_c1, t_e1, t_e1_stack, t_e1b, zero_val_unet_c1)
    i_unet_c1 = n_e1_mid_unet_c1 + (__jacc_i - 1) * -1
    __idx_t_e1_stack_0 = (i_unet_c1 - 1) + 1
    t_e1[i_unet_c1] = t_e1_stack[__idx_t_e1_stack_0]
    __oldb_2 = t_e1b[i_unet_c1]
    t_e1b[i_unet_c1] = 0.0
    t_e1b[i_unet_c1] = t_e1b[i_unet_c1] + (0.5 * (1.0 + sign(t_e1[i_unet_c1] - zero_val_unet_c1))) * __oldb_2
    return nothing
end

function jacc_kernel_unet_loss_b_99!(__jacc_i, b_e1ab, c1, c_in, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_e1b, w, w_e1a, w_e1ab, wp1_unet_c1, xpad0, xpad0b)
    idx_unet_c1 = c1 * hw_unet_c1 + (__jacc_i - 1) * -1
    s_unet_c1b = 0.0
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    __oldb_0 = t_e1b[idx_unet_c1]
    t_e1b[idx_unet_c1] = 0.0
    s_unet_c1b = s_unet_c1b + __oldb_0
    Atomix.@atomic b_e1ab[co_unet_c1] += __oldb_0
    for i_k_unet_c1 = c_in * khkw_unet_c1:-1:1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        __icse_44 = ci_unet_c1 - 1
        xi_unet_c1 = __icse_44 * hp1_unet_c1 * wp1_unet_c1 + (row_unet_c1 - 1) * wp1_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c_in + __icse_44) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        Atomix.@atomic xpad0b[xi_unet_c1] += w_e1a[wi_unet_c1] * s_unet_c1b
        Atomix.@atomic w_e1ab[wi_unet_c1] += xpad0[xi_unet_c1] * s_unet_c1b
    end
    s_unet_c1b = 0.0
    return nothing
end

function jacc_kernel_unet_loss_b_100!(__jacc_i, c_in, hp1_unet_c1, hw_unet_c1, pad_unet_c1, w, wp1_unet_c1, xb, xpad0b)
    idx_unet_c1 = c_in * hw_unet_c1 + (__jacc_i - 1) * -1
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp1_unet_c1 * wp1_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp1_unet_c1 + (j_unet_c1 + pad_unet_c1)
    __oldb_0 = xpad0b[yi_unet_c1]
    xpad0b[yi_unet_c1] = 0.0
    xb[idx_unet_c1] = xb[idx_unet_c1] + __oldb_0
    return nothing
end

function jacc_kernel_unet_loss_b_101!(__jacc_i, n_xpad0_unet_c1, xpad0b)
    i_unet_c1 = n_xpad0_unet_c1 + (__jacc_i - 1) * -1
    xpad0b[i_unet_c1] = 0.0
    return nothing
end

function jacc_kernel_unet_loss_1!(__jacc_i, n_xpad0_unet_c1, xpad0, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    xpad0[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_2!(__jacc_i, c_in, hp1_unet_c1, hw_unet_c1, pad_unet_c1, w, wp1_unet_c1, x, xpad0)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp1_unet_c1 * wp1_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp1_unet_c1 + (j_unet_c1 + pad_unet_c1)
    xpad0[yi_unet_c1] = x[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_3!(__jacc_i, b_e1a, c1, c_in, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_e1, w, w_e1a, wp1_unet_c1, xpad0)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c_in * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        xi_unet_c1 = (ci_unet_c1 - 1) * hp1_unet_c1 * wp1_unet_c1 + (row_unet_c1 - 1) * wp1_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c_in + (ci_unet_c1 - 1)) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + xpad0[xi_unet_c1] * w_e1a[wi_unet_c1]
    end
    t_e1[idx_unet_c1] = s_unet_c1 + b_e1a[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_4!(__jacc_i, n_e1_mid_unet_c1, t_e1, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    t_e1[i_unet_c1] = max(t_e1[i_unet_c1], zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_5!(__jacc_i, n_e1_midpad_unet_c1, t_e1pad, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    t_e1pad[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_6!(__jacc_i, c1, hp1_unet_c1, hw_unet_c1, pad_unet_c1, t_e1, t_e1pad, w, wp1_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp1_unet_c1 * wp1_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp1_unet_c1 + (j_unet_c1 + pad_unet_c1)
    t_e1pad[yi_unet_c1] = t_e1[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_7!(__jacc_i, b_e1b, c1, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, skip1, t_e1pad, w, w_e1b, wp1_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c1 * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        xi_unet_c1 = (ci_unet_c1 - 1) * hp1_unet_c1 * wp1_unet_c1 + (row_unet_c1 - 1) * wp1_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c1 + (ci_unet_c1 - 1)) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + t_e1pad[xi_unet_c1] * w_e1b[wi_unet_c1]
    end
    skip1[idx_unet_c1] = s_unet_c1 + b_e1b[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_8!(__jacc_i, n_e1_out_unet_c1, skip1, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    skip1[i_unet_c1] = max(skip1[i_unet_c1], zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_9!(__jacc_i, c1, hw2_unet_c1, hw_unet_c1, p1, skip1, w, w2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    oi_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    oj_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    i0_unet_c1 = 2oi_unet_c1 - 1
    j0_unet_c1 = 2oj_unet_c1 - 1
    a11_unet_c1 = skip1[(ci_unet_c1 - 1) * hw_unet_c1 + (i0_unet_c1 - 1) * w + j0_unet_c1]
    a12_unet_c1 = skip1[(ci_unet_c1 - 1) * hw_unet_c1 + (i0_unet_c1 - 1) * w + j0_unet_c1 + 1]
    a21_unet_c1 = skip1[(ci_unet_c1 - 1) * hw_unet_c1 + i0_unet_c1 * w + j0_unet_c1]
    a22_unet_c1 = skip1[(ci_unet_c1 - 1) * hw_unet_c1 + i0_unet_c1 * w + j0_unet_c1 + 1]
    m1_unet_c1 = max(a11_unet_c1, a12_unet_c1)
    m2_unet_c1 = max(a21_unet_c1, a22_unet_c1)
    p1[idx_unet_c1] = max(m1_unet_c1, m2_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_10!(__jacc_i, n_p1pad_unet_c1, p1pad, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    p1pad[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_11!(__jacc_i, c1, hp2_unet_c1, hw2_unet_c1, p1, p1pad, pad_unet_c1, w2_unet_c1, wp2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp2_unet_c1 * wp2_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp2_unet_c1 + (j_unet_c1 + pad_unet_c1)
    p1pad[yi_unet_c1] = p1[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_12!(__jacc_i, b_e2a, c1, c2, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, p1pad, t_e2, w2_unet_c1, w_e2a, wp2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c1 * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        xi_unet_c1 = (ci_unet_c1 - 1) * hp2_unet_c1 * wp2_unet_c1 + (row_unet_c1 - 1) * wp2_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c1 + (ci_unet_c1 - 1)) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + p1pad[xi_unet_c1] * w_e2a[wi_unet_c1]
    end
    t_e2[idx_unet_c1] = s_unet_c1 + b_e2a[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_13!(__jacc_i, n_e2_mid_unet_c1, t_e2, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    t_e2[i_unet_c1] = max(t_e2[i_unet_c1], zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_14!(__jacc_i, n_e2_midpad_unet_c1, t_e2pad, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    t_e2pad[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_15!(__jacc_i, c2, hp2_unet_c1, hw2_unet_c1, pad_unet_c1, t_e2, t_e2pad, w2_unet_c1, wp2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp2_unet_c1 * wp2_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp2_unet_c1 + (j_unet_c1 + pad_unet_c1)
    t_e2pad[yi_unet_c1] = t_e2[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_16!(__jacc_i, b_e2b, c2, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, skip2, t_e2pad, w2_unet_c1, w_e2b, wp2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c2 * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        xi_unet_c1 = (ci_unet_c1 - 1) * hp2_unet_c1 * wp2_unet_c1 + (row_unet_c1 - 1) * wp2_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c2 + (ci_unet_c1 - 1)) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + t_e2pad[xi_unet_c1] * w_e2b[wi_unet_c1]
    end
    skip2[idx_unet_c1] = s_unet_c1 + b_e2b[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_17!(__jacc_i, n_e2_out_unet_c1, skip2, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    skip2[i_unet_c1] = max(skip2[i_unet_c1], zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_18!(__jacc_i, c2, hw2_unet_c1, hw4_unet_c1, p2, skip2, w2_unet_c1, w4_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw4_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw4_unet_c1)
    oi_unet_c1 = div(rem_unet_c1, w4_unet_c1) + 1
    oj_unet_c1 = mod(rem_unet_c1, w4_unet_c1) + 1
    i0_unet_c1 = 2oi_unet_c1 - 1
    j0_unet_c1 = 2oj_unet_c1 - 1
    a11_unet_c1 = skip2[(ci_unet_c1 - 1) * hw2_unet_c1 + (i0_unet_c1 - 1) * w2_unet_c1 + j0_unet_c1]
    a12_unet_c1 = skip2[(ci_unet_c1 - 1) * hw2_unet_c1 + (i0_unet_c1 - 1) * w2_unet_c1 + j0_unet_c1 + 1]
    a21_unet_c1 = skip2[(ci_unet_c1 - 1) * hw2_unet_c1 + i0_unet_c1 * w2_unet_c1 + j0_unet_c1]
    a22_unet_c1 = skip2[(ci_unet_c1 - 1) * hw2_unet_c1 + i0_unet_c1 * w2_unet_c1 + j0_unet_c1 + 1]
    m1_unet_c1 = max(a11_unet_c1, a12_unet_c1)
    m2_unet_c1 = max(a21_unet_c1, a22_unet_c1)
    p2[idx_unet_c1] = max(m1_unet_c1, m2_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_19!(__jacc_i, n_p2pad_unet_c1, p2pad, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    p2pad[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_20!(__jacc_i, c2, hp4_unet_c1, hw4_unet_c1, p2, p2pad, pad_unet_c1, w4_unet_c1, wp4_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw4_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw4_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w4_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w4_unet_c1) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp4_unet_c1 * wp4_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp4_unet_c1 + (j_unet_c1 + pad_unet_c1)
    p2pad[yi_unet_c1] = p2[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_21!(__jacc_i, b_ba, c2, c3, hp4_unet_c1, hw4_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, p2pad, t_b, w4_unet_c1, w_ba, wp4_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw4_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw4_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w4_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w4_unet_c1) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c2 * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        xi_unet_c1 = (ci_unet_c1 - 1) * hp4_unet_c1 * wp4_unet_c1 + (row_unet_c1 - 1) * wp4_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c2 + (ci_unet_c1 - 1)) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + p2pad[xi_unet_c1] * w_ba[wi_unet_c1]
    end
    t_b[idx_unet_c1] = s_unet_c1 + b_ba[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_22!(__jacc_i, n_b_mid_unet_c1, t_b, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    t_b[i_unet_c1] = max(t_b[i_unet_c1], zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_23!(__jacc_i, n_b_midpad_unet_c1, t_bpad, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    t_bpad[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_24!(__jacc_i, c3, hp4_unet_c1, hw4_unet_c1, pad_unet_c1, t_b, t_bpad, w4_unet_c1, wp4_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw4_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw4_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w4_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w4_unet_c1) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp4_unet_c1 * wp4_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp4_unet_c1 + (j_unet_c1 + pad_unet_c1)
    t_bpad[yi_unet_c1] = t_b[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_25!(__jacc_i, b_bb, bott, c3, hp4_unet_c1, hw4_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_bpad, w4_unet_c1, w_bb, wp4_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw4_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw4_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w4_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w4_unet_c1) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c3 * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        xi_unet_c1 = (ci_unet_c1 - 1) * hp4_unet_c1 * wp4_unet_c1 + (row_unet_c1 - 1) * wp4_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c3 + (ci_unet_c1 - 1)) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + t_bpad[xi_unet_c1] * w_bb[wi_unet_c1]
    end
    bott[idx_unet_c1] = s_unet_c1 + b_bb[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_26!(__jacc_i, bott, n_b_out_unet_c1, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    bott[i_unet_c1] = max(bott[i_unet_c1], zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_27!(__jacc_i, bott, c3, hw2_unet_c1, hw4_unet_c1, scale_unet_c1, u2, w2_unet_c1, w4_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    oi_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    oj_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    oim1_unet_c1 = oi_unet_c1 - 1
    ojm1_unet_c1 = oj_unet_c1 - 1
    i_unet_c1 = div(oim1_unet_c1, scale_unet_c1) + 1
    j_unet_c1 = div(ojm1_unet_c1, scale_unet_c1) + 1
    xi_unet_c1 = (ci_unet_c1 - 1) * hw4_unet_c1 + (i_unet_c1 - 1) * w4_unet_c1 + j_unet_c1
    u2[idx_unet_c1] = bott[xi_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_28!(__jacc_i, c3, cat2, hw2_unet_c1, u2)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    cat2[idx_unet_c1] = u2[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_29!(__jacc_i, c2, c3, cat2, hw2_unet_c1, skip2)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    cat2[c3 * hw2_unet_c1 + idx_unet_c1] = skip2[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_30!(__jacc_i, cat2pad, n_cat2pad_unet_c1, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    cat2pad[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_31!(__jacc_i, c32_unet_c1, cat2, cat2pad, hp2_unet_c1, hw2_unet_c1, pad_unet_c1, w2_unet_c1, wp2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp2_unet_c1 * wp2_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp2_unet_c1 + (j_unet_c1 + pad_unet_c1)
    cat2pad[yi_unet_c1] = cat2[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_32!(__jacc_i, b_d2a, c2, c32_unet_c1, cat2pad, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d2, w2_unet_c1, w_d2a, wp2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c32_unet_c1 * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        xi_unet_c1 = (ci_unet_c1 - 1) * hp2_unet_c1 * wp2_unet_c1 + (row_unet_c1 - 1) * wp2_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c32_unet_c1 + (ci_unet_c1 - 1)) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + cat2pad[xi_unet_c1] * w_d2a[wi_unet_c1]
    end
    t_d2[idx_unet_c1] = s_unet_c1 + b_d2a[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_33!(__jacc_i, n_d2_mid_unet_c1, t_d2, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    t_d2[i_unet_c1] = max(t_d2[i_unet_c1], zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_34!(__jacc_i, n_d2_midpad_unet_c1, t_d2pad, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    t_d2pad[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_35!(__jacc_i, c2, hp2_unet_c1, hw2_unet_c1, pad_unet_c1, t_d2, t_d2pad, w2_unet_c1, wp2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp2_unet_c1 * wp2_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp2_unet_c1 + (j_unet_c1 + pad_unet_c1)
    t_d2pad[yi_unet_c1] = t_d2[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_36!(__jacc_i, b_d2b, c2, dec2out, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d2pad, w2_unet_c1, w_d2b, wp2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw2_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw2_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w2_unet_c1) + 1
    j_unet_c1 = mod(rem_unet_c1, w2_unet_c1) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c2 * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        xi_unet_c1 = (ci_unet_c1 - 1) * hp2_unet_c1 * wp2_unet_c1 + (row_unet_c1 - 1) * wp2_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c2 + (ci_unet_c1 - 1)) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + t_d2pad[xi_unet_c1] * w_d2b[wi_unet_c1]
    end
    dec2out[idx_unet_c1] = s_unet_c1 + b_d2b[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_37!(__jacc_i, dec2out, n_d2_out_unet_c1, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    dec2out[i_unet_c1] = max(dec2out[i_unet_c1], zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_38!(__jacc_i, c2, dec2out, hw2_unet_c1, hw_unet_c1, scale_unet_c1, u1, w, w2_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    oi_unet_c1 = div(rem_unet_c1, w) + 1
    oj_unet_c1 = mod(rem_unet_c1, w) + 1
    oim1_unet_c1 = oi_unet_c1 - 1
    ojm1_unet_c1 = oj_unet_c1 - 1
    i_unet_c1 = div(oim1_unet_c1, scale_unet_c1) + 1
    j_unet_c1 = div(ojm1_unet_c1, scale_unet_c1) + 1
    xi_unet_c1 = (ci_unet_c1 - 1) * hw2_unet_c1 + (i_unet_c1 - 1) * w2_unet_c1 + j_unet_c1
    u1[idx_unet_c1] = dec2out[xi_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_39!(__jacc_i, c2, cat1, hw_unet_c1, u1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    cat1[idx_unet_c1] = u1[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_40!(__jacc_i, c1, c2, cat1, hw_unet_c1, skip1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    cat1[c2 * hw_unet_c1 + idx_unet_c1] = skip1[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_41!(__jacc_i, cat1pad, n_cat1pad_unet_c1, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    cat1pad[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_42!(__jacc_i, c21_unet_c1, cat1, cat1pad, hp1_unet_c1, hw_unet_c1, pad_unet_c1, w, wp1_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp1_unet_c1 * wp1_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp1_unet_c1 + (j_unet_c1 + pad_unet_c1)
    cat1pad[yi_unet_c1] = cat1[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_43!(__jacc_i, b_d1a, c1, c21_unet_c1, cat1pad, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d1, w, w_d1a, wp1_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c21_unet_c1 * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        xi_unet_c1 = (ci_unet_c1 - 1) * hp1_unet_c1 * wp1_unet_c1 + (row_unet_c1 - 1) * wp1_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c21_unet_c1 + (ci_unet_c1 - 1)) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + cat1pad[xi_unet_c1] * w_d1a[wi_unet_c1]
    end
    t_d1[idx_unet_c1] = s_unet_c1 + b_d1a[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_44!(__jacc_i, n_d1_mid_unet_c1, t_d1, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    t_d1[i_unet_c1] = max(t_d1[i_unet_c1], zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_45!(__jacc_i, n_d1_midpad_unet_c1, t_d1pad, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    t_d1pad[i_unet_c1] = zero_val_unet_c1
    return nothing
end

function jacc_kernel_unet_loss_46!(__jacc_i, c1, hp1_unet_c1, hw_unet_c1, pad_unet_c1, t_d1, t_d1pad, w, wp1_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    ci_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    yi_unet_c1 = (ci_unet_c1 - 1) * hp1_unet_c1 * wp1_unet_c1 + ((i_unet_c1 + pad_unet_c1) - 1) * wp1_unet_c1 + (j_unet_c1 + pad_unet_c1)
    t_d1pad[yi_unet_c1] = t_d1[idx_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_47!(__jacc_i, b_d1b, c1, dec1out, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d1pad, w, w_d1b, wp1_unet_c1)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c1 * khkw_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        xi_unet_c1 = (ci_unet_c1 - 1) * hp1_unet_c1 * wp1_unet_c1 + (row_unet_c1 - 1) * wp1_unet_c1 + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c1 + (ci_unet_c1 - 1)) * kh_unet_c1 + (ki_unet_c1 - 1)) * kw_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + t_d1pad[xi_unet_c1] * w_d1b[wi_unet_c1]
    end
    dec1out[idx_unet_c1] = s_unet_c1 + b_d1b[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_48!(__jacc_i, dec1out, n_d1_out_unet_c1, zero_val_unet_c1)
    i_unet_c1 = 1 + (__jacc_i - 1)
    dec1out[i_unet_c1] = max(dec1out[i_unet_c1], zero_val_unet_c1)
    return nothing
end

function jacc_kernel_unet_loss_49!(__jacc_i, b_out, c1, c_out, dec1out, hw_unet_c1, kh_out_unet_c1, khkw_out_unet_c1, kw_out_unet_c1, w, w_out, y)
    idx_unet_c1 = 1 + (__jacc_i - 1)
    idxm1_unet_c1 = idx_unet_c1 - 1
    co_unet_c1 = div(idxm1_unet_c1, hw_unet_c1) + 1
    rem_unet_c1 = mod(idxm1_unet_c1, hw_unet_c1)
    i_unet_c1 = div(rem_unet_c1, w) + 1
    j_unet_c1 = mod(rem_unet_c1, w) + 1
    s_unet_c1 = 0.0
    for i_k_unet_c1 = 1:c1 * khkw_out_unet_c1
        kseqm1_unet_c1 = i_k_unet_c1 - 1
        ci_unet_c1 = div(kseqm1_unet_c1, khkw_out_unet_c1) + 1
        rem2_unet_c1 = mod(kseqm1_unet_c1, khkw_out_unet_c1)
        ki_unet_c1 = div(rem2_unet_c1, kw_out_unet_c1) + 1
        kj_unet_c1 = mod(rem2_unet_c1, kw_out_unet_c1) + 1
        row_unet_c1 = (i_unet_c1 + ki_unet_c1) - 1
        col_unet_c1 = (j_unet_c1 + kj_unet_c1) - 1
        xi_unet_c1 = (ci_unet_c1 - 1) * hw_unet_c1 + (row_unet_c1 - 1) * w + col_unet_c1
        wi_unet_c1 = (((co_unet_c1 - 1) * c1 + (ci_unet_c1 - 1)) * kh_out_unet_c1 + (ki_unet_c1 - 1)) * kw_out_unet_c1 + kj_unet_c1
        s_unet_c1 = s_unet_c1 + dec1out[xi_unet_c1] * w_out[wi_unet_c1]
    end
    y[idx_unet_c1] = s_unet_c1 + b_out[co_unet_c1]
    return nothing
end

function jacc_kernel_unet_loss_50!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_unet_loss_51!(__jacc_i, c_out, h, loss, target, w, y)
    i_o = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += (y[i_o] - target[i_o]) ^ 2
    return nothing
end

function initstacks_unet_loss_b_jacc(c1, c2, c3, h, w)
    two_unet_c1 = 2
    h2_unet_c1 = div(h, two_unet_c1)
    w2_unet_c1 = div(w, two_unet_c1)
    hw2_unet_c1 = h2_unet_c1 * w2_unet_c1
    four_unet_c1 = 4
    w4_unet_c1 = div(w, four_unet_c1)
    h4_unet_c1 = div(h, four_unet_c1)
    hw4_unet_c1 = h4_unet_c1 * w4_unet_c1
    n_b_mid_unet_c1 = c3 * hw4_unet_c1
    n_b_out_unet_c1 = c3 * hw4_unet_c1
    hw_unet_c1 = h * w
    n_d1_mid_unet_c1 = c1 * hw_unet_c1
    n_d1_out_unet_c1 = c1 * hw_unet_c1
    n_d2_mid_unet_c1 = c2 * hw2_unet_c1
    n_d2_out_unet_c1 = c2 * hw2_unet_c1
    n_e1_mid_unet_c1 = c1 * hw_unet_c1
    n_e1_out_unet_c1 = c1 * hw_unet_c1
    n_e2_mid_unet_c1 = c2 * hw2_unet_c1
    n_e2_out_unet_c1 = c2 * hw2_unet_c1
    t_e1_stack = JACC.zeros(Float64, max(0, div(n_e1_mid_unet_c1 - 1, 1) + 1))
    skip1_stack = JACC.zeros(Float64, max(0, div(n_e1_out_unet_c1 - 1, 1) + 1))
    t_e2_stack = JACC.zeros(Float64, max(0, div(n_e2_mid_unet_c1 - 1, 1) + 1))
    skip2_stack = JACC.zeros(Float64, max(0, div(n_e2_out_unet_c1 - 1, 1) + 1))
    t_b_stack = JACC.zeros(Float64, max(0, div(n_b_mid_unet_c1 - 1, 1) + 1))
    bott_stack = JACC.zeros(Float64, max(0, div(n_b_out_unet_c1 - 1, 1) + 1))
    t_d2_stack = JACC.zeros(Float64, max(0, div(n_d2_mid_unet_c1 - 1, 1) + 1))
    dec2out_stack = JACC.zeros(Float64, max(0, div(n_d2_out_unet_c1 - 1, 1) + 1))
    t_d1_stack = JACC.zeros(Float64, max(0, div(n_d1_mid_unet_c1 - 1, 1) + 1))
    dec1out_stack = JACC.zeros(Float64, max(0, div(n_d1_out_unet_c1 - 1, 1) + 1))
    return (t_e1_stack, skip1_stack, t_e2_stack, skip2_stack, t_b_stack, bott_stack, t_d2_stack, dec2out_stack, t_d1_stack, dec1out_stack)
end

function unet_loss_b_jacc(x, xb, h, w, c_in, c1, c2, c3, c_out, w_e1a, w_e1ab, b_e1a, b_e1ab, w_e1b, w_e1bb, b_e1b, b_e1bb, w_e2a, w_e2ab, b_e2a, b_e2ab, w_e2b, w_e2bb, b_e2b, b_e2bb, w_ba, w_bab, b_ba, b_bab, w_bb, w_bbb, b_bb, b_bbb, w_d2a, w_d2ab, b_d2a, b_d2ab, w_d2b, w_d2bb, b_d2b, b_d2bb, w_d1a, w_d1ab, b_d1a, b_d1ab, w_d1b, w_d1bb, b_d1b, b_d1bb, w_out, w_outb, b_out, b_outb, xpad0, xpad0b, t_e1, t_e1b, t_e1pad, t_e1padb, skip1, skip1b, p1, p1b, p1pad, p1padb, t_e2, t_e2b, t_e2pad, t_e2padb, skip2, skip2b, p2, p2b, p2pad, p2padb, t_b, t_bb, t_bpad, t_bpadb, bott, bottb, u2, u2b, cat2, cat2b, cat2pad, cat2padb, t_d2, t_d2b, t_d2pad, t_d2padb, dec2out, dec2outb, u1, u1b, cat1, cat1b, cat1pad, cat1padb, t_d1, t_d1b, t_d1pad, t_d1padb, dec1out, dec1outb, y, yb, target, targetb, loss, lossb, t_e1_stack, skip1_stack, t_e2_stack, skip2_stack, t_b_stack, bott_stack, t_d2_stack, dec2out_stack, t_d1_stack, dec1out_stack)
    a11_unet_c1 = 0.0
    a12_unet_c1 = 0.0
    a21_unet_c1 = 0.0
    a22_unet_c1 = 0.0
    m1_unet_c1 = 0.0
    m2_unet_c1 = 0.0
    s_unet_c1 = 0.0
    a11_unet_c1b = 0.0
    a12_unet_c1b = 0.0
    a21_unet_c1b = 0.0
    a22_unet_c1b = 0.0
    m1_unet_c1b = 0.0
    m2_unet_c1b = 0.0
    s_unet_c1b = 0.0
    two_unet_c1 = 2
    four_unet_c1 = 4
    h2_unet_c1 = div(h, two_unet_c1)
    w2_unet_c1 = div(w, two_unet_c1)
    h4_unet_c1 = div(h, four_unet_c1)
    w4_unet_c1 = div(w, four_unet_c1)
    hp1_unet_c1 = h + 2
    wp1_unet_c1 = w + 2
    hp2_unet_c1 = h2_unet_c1 + 2
    wp2_unet_c1 = w2_unet_c1 + 2
    hp4_unet_c1 = h4_unet_c1 + 2
    wp4_unet_c1 = w4_unet_c1 + 2
    pad_unet_c1 = 1
    kh_unet_c1 = 3
    kw_unet_c1 = 3
    khkw_unet_c1 = kh_unet_c1 * kw_unet_c1
    kh_out_unet_c1 = 1
    kw_out_unet_c1 = 1
    khkw_out_unet_c1 = kh_out_unet_c1 * kw_out_unet_c1
    scale_unet_c1 = 2
    zero_val_unet_c1 = 0.0
    c32_unet_c1 = c3 + c2
    c21_unet_c1 = c2 + c1
    hw_unet_c1 = h * w
    hw2_unet_c1 = h2_unet_c1 * w2_unet_c1
    hw4_unet_c1 = h4_unet_c1 * w4_unet_c1
    n_xpad0_unet_c1 = c_in * hp1_unet_c1 * wp1_unet_c1
    __icse_0 = c1 * hw_unet_c1
    n_e1_mid_unet_c1 = __icse_0
    __icse_1 = c1 * hp1_unet_c1 * wp1_unet_c1
    n_e1_midpad_unet_c1 = __icse_1
    n_e1_out_unet_c1 = __icse_0
    n_p1pad_unet_c1 = c1 * hp2_unet_c1 * wp2_unet_c1
    __icse_2 = c2 * hw2_unet_c1
    n_e2_mid_unet_c1 = __icse_2
    __icse_3 = c2 * hp2_unet_c1 * wp2_unet_c1
    n_e2_midpad_unet_c1 = __icse_3
    n_e2_out_unet_c1 = __icse_2
    n_p2pad_unet_c1 = c2 * hp4_unet_c1 * wp4_unet_c1
    __icse_4 = c3 * hw4_unet_c1
    n_b_mid_unet_c1 = __icse_4
    n_b_midpad_unet_c1 = c3 * hp4_unet_c1 * wp4_unet_c1
    n_b_out_unet_c1 = __icse_4
    n_cat2pad_unet_c1 = c32_unet_c1 * hp2_unet_c1 * wp2_unet_c1
    n_d2_mid_unet_c1 = __icse_2
    n_d2_midpad_unet_c1 = __icse_3
    n_d2_out_unet_c1 = __icse_2
    n_cat1pad_unet_c1 = c21_unet_c1 * hp1_unet_c1 * wp1_unet_c1
    n_d1_mid_unet_c1 = __icse_0
    n_d1_midpad_unet_c1 = __icse_1
    n_d1_out_unet_c1 = __icse_0
    if div(n_xpad0_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_xpad0_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_1!(n_xpad0_unet_c1, xpad0, zero_val_unet_c1)
    end
    if div(c_in * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c_in * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_2!(c_in, hp1_unet_c1, hw_unet_c1, pad_unet_c1, w, wp1_unet_c1, x, xpad0)
    end
    if div(c1 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_3!(b_e1a, c1, c_in, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_e1, w, w_e1a, wp1_unet_c1, xpad0)
    end
    if div(n_e1_mid_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_e1_mid_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_4!(n_e1_mid_unet_c1, t_e1, t_e1_stack, zero_val_unet_c1)
    end
    if div(n_e1_midpad_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_e1_midpad_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_5!(n_e1_midpad_unet_c1, t_e1pad, zero_val_unet_c1)
    end
    if div(c1 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_6!(c1, hp1_unet_c1, hw_unet_c1, pad_unet_c1, t_e1, t_e1pad, w, wp1_unet_c1)
    end
    if div(c1 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_7!(b_e1b, c1, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, skip1, t_e1pad, w, w_e1b, wp1_unet_c1)
    end
    if div(n_e1_out_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_e1_out_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_8!(n_e1_out_unet_c1, skip1, skip1_stack, zero_val_unet_c1)
    end
    if div(c1 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_9!(c1, hw2_unet_c1, hw_unet_c1, p1, skip1, w, w2_unet_c1)
    end
    if div(n_p1pad_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_p1pad_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_10!(n_p1pad_unet_c1, p1pad, zero_val_unet_c1)
    end
    if div(c1 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_11!(c1, hp2_unet_c1, hw2_unet_c1, p1, p1pad, pad_unet_c1, w2_unet_c1, wp2_unet_c1)
    end
    if div(c2 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_12!(b_e2a, c1, c2, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, p1pad, t_e2, w2_unet_c1, w_e2a, wp2_unet_c1)
    end
    if div(n_e2_mid_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_e2_mid_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_13!(n_e2_mid_unet_c1, t_e2, t_e2_stack, zero_val_unet_c1)
    end
    if div(n_e2_midpad_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_e2_midpad_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_14!(n_e2_midpad_unet_c1, t_e2pad, zero_val_unet_c1)
    end
    if div(c2 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_15!(c2, hp2_unet_c1, hw2_unet_c1, pad_unet_c1, t_e2, t_e2pad, w2_unet_c1, wp2_unet_c1)
    end
    if div(c2 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_16!(b_e2b, c2, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, skip2, t_e2pad, w2_unet_c1, w_e2b, wp2_unet_c1)
    end
    if div(n_e2_out_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_e2_out_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_17!(n_e2_out_unet_c1, skip2, skip2_stack, zero_val_unet_c1)
    end
    if div(c2 * hw4_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw4_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_18!(c2, hw2_unet_c1, hw4_unet_c1, p2, skip2, w2_unet_c1, w4_unet_c1)
    end
    if div(n_p2pad_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_p2pad_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_19!(n_p2pad_unet_c1, p2pad, zero_val_unet_c1)
    end
    if div(c2 * hw4_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw4_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_20!(c2, hp4_unet_c1, hw4_unet_c1, p2, p2pad, pad_unet_c1, w4_unet_c1, wp4_unet_c1)
    end
    if div(c3 * hw4_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c3 * hw4_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_21!(b_ba, c2, c3, hp4_unet_c1, hw4_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, p2pad, t_b, w4_unet_c1, w_ba, wp4_unet_c1)
    end
    if div(n_b_mid_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_b_mid_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_22!(n_b_mid_unet_c1, t_b, t_b_stack, zero_val_unet_c1)
    end
    if div(n_b_midpad_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_b_midpad_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_23!(n_b_midpad_unet_c1, t_bpad, zero_val_unet_c1)
    end
    if div(c3 * hw4_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c3 * hw4_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_24!(c3, hp4_unet_c1, hw4_unet_c1, pad_unet_c1, t_b, t_bpad, w4_unet_c1, wp4_unet_c1)
    end
    if div(c3 * hw4_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c3 * hw4_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_25!(b_bb, bott, c3, hp4_unet_c1, hw4_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_bpad, w4_unet_c1, w_bb, wp4_unet_c1)
    end
    if div(n_b_out_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_b_out_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_26!(bott, bott_stack, n_b_out_unet_c1, zero_val_unet_c1)
    end
    if div(c3 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c3 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_27!(bott, c3, hw2_unet_c1, hw4_unet_c1, scale_unet_c1, u2, w2_unet_c1, w4_unet_c1)
    end
    if div(c3 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c3 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_28!(c3, cat2, hw2_unet_c1, u2)
    end
    if div(c2 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_29!(c2, c3, cat2, hw2_unet_c1, skip2)
    end
    if div(n_cat2pad_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_cat2pad_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_30!(cat2pad, n_cat2pad_unet_c1, zero_val_unet_c1)
    end
    if div(c32_unet_c1 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c32_unet_c1 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_31!(c32_unet_c1, cat2, cat2pad, hp2_unet_c1, hw2_unet_c1, pad_unet_c1, w2_unet_c1, wp2_unet_c1)
    end
    if div(c2 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_32!(b_d2a, c2, c32_unet_c1, cat2pad, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d2, w2_unet_c1, w_d2a, wp2_unet_c1)
    end
    if div(n_d2_mid_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_d2_mid_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_33!(n_d2_mid_unet_c1, t_d2, t_d2_stack, zero_val_unet_c1)
    end
    if div(n_d2_midpad_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_d2_midpad_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_34!(n_d2_midpad_unet_c1, t_d2pad, zero_val_unet_c1)
    end
    if div(c2 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_35!(c2, hp2_unet_c1, hw2_unet_c1, pad_unet_c1, t_d2, t_d2pad, w2_unet_c1, wp2_unet_c1)
    end
    if div(c2 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_36!(b_d2b, c2, dec2out, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d2pad, w2_unet_c1, w_d2b, wp2_unet_c1)
    end
    if div(n_d2_out_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_d2_out_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_37!(dec2out, dec2out_stack, n_d2_out_unet_c1, zero_val_unet_c1)
    end
    if div(c2 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_38!(c2, dec2out, hw2_unet_c1, hw_unet_c1, scale_unet_c1, u1, w, w2_unet_c1)
    end
    if div(c2 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_39!(c2, cat1, hw_unet_c1, u1)
    end
    if div(c1 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_40!(c1, c2, cat1, hw_unet_c1, skip1)
    end
    if div(n_cat1pad_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_cat1pad_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_41!(cat1pad, n_cat1pad_unet_c1, zero_val_unet_c1)
    end
    if div(c21_unet_c1 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c21_unet_c1 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_42!(c21_unet_c1, cat1, cat1pad, hp1_unet_c1, hw_unet_c1, pad_unet_c1, w, wp1_unet_c1)
    end
    if div(c1 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_43!(b_d1a, c1, c21_unet_c1, cat1pad, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d1, w, w_d1a, wp1_unet_c1)
    end
    if div(n_d1_mid_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_d1_mid_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_44!(n_d1_mid_unet_c1, t_d1, t_d1_stack, zero_val_unet_c1)
    end
    if div(n_d1_midpad_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_d1_midpad_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_45!(n_d1_midpad_unet_c1, t_d1pad, zero_val_unet_c1)
    end
    if div(c1 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_46!(c1, hp1_unet_c1, hw_unet_c1, pad_unet_c1, t_d1, t_d1pad, w, wp1_unet_c1)
    end
    if div(c1 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_47!(b_d1b, c1, dec1out, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d1pad, w, w_d1b, wp1_unet_c1)
    end
    if div(n_d1_out_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_d1_out_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_48!(dec1out, dec1out_stack, n_d1_out_unet_c1, zero_val_unet_c1)
    end
    if div(c_out * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c_out * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_49!(b_out, c1, c_out, dec1out, hw_unet_c1, kh_out_unet_c1, khkw_out_unet_c1, kw_out_unet_c1, w, w_out, y)
    end
    if div(c_out * h * w - 1, 1) + 1 < 32768
        if div(c_out * h * w - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(c_out * h * w - 1, 1) + 1 jacc_kernel_unet_loss_b_51!(c_out, h, loss, target, w, y)
        end
    else
        if div(c_out * h * w - 1, 1) + 1 > 0
            __jgen_redval_50 = JACC.@parallel_reduce(range = div(c_out * h * w - 1, 1) + 1, (((i_o, target, y)->(y[i_o] - target[i_o]) ^ 2))(target, y))
            JACC.@parallel_for range = 1 jacc_kernel_unet_loss_b_50!(loss, __jgen_redval_50)
        end
    end
    two_unet_c1 = 2
    four_unet_c1 = 4
    h2_unet_c1 = div(h, two_unet_c1)
    w2_unet_c1 = div(w, two_unet_c1)
    h4_unet_c1 = div(h, four_unet_c1)
    w4_unet_c1 = div(w, four_unet_c1)
    hp1_unet_c1 = h + 2
    wp1_unet_c1 = w + 2
    hp2_unet_c1 = h2_unet_c1 + 2
    wp2_unet_c1 = w2_unet_c1 + 2
    hp4_unet_c1 = h4_unet_c1 + 2
    wp4_unet_c1 = w4_unet_c1 + 2
    pad_unet_c1 = 1
    kh_unet_c1 = 3
    kw_unet_c1 = 3
    khkw_unet_c1 = kh_unet_c1 * kw_unet_c1
    kh_out_unet_c1 = 1
    kw_out_unet_c1 = 1
    khkw_out_unet_c1 = kh_out_unet_c1 * kw_out_unet_c1
    scale_unet_c1 = 2
    c32_unet_c1 = c3 + c2
    c21_unet_c1 = c2 + c1
    hw_unet_c1 = h * w
    hw2_unet_c1 = h2_unet_c1 * w2_unet_c1
    hw4_unet_c1 = h4_unet_c1 * w4_unet_c1
    n_xpad0_unet_c1 = c_in * hp1_unet_c1 * wp1_unet_c1
    __icse_26 = c1 * hw_unet_c1
    n_e1_mid_unet_c1 = __icse_26
    __icse_27 = c1 * hp1_unet_c1 * wp1_unet_c1
    n_e1_midpad_unet_c1 = __icse_27
    n_e1_out_unet_c1 = __icse_26
    n_p1pad_unet_c1 = c1 * hp2_unet_c1 * wp2_unet_c1
    __icse_28 = c2 * hw2_unet_c1
    n_e2_mid_unet_c1 = __icse_28
    __icse_29 = c2 * hp2_unet_c1 * wp2_unet_c1
    n_e2_midpad_unet_c1 = __icse_29
    n_e2_out_unet_c1 = __icse_28
    n_p2pad_unet_c1 = c2 * hp4_unet_c1 * wp4_unet_c1
    __icse_30 = c3 * hw4_unet_c1
    n_b_mid_unet_c1 = __icse_30
    n_b_midpad_unet_c1 = c3 * hp4_unet_c1 * wp4_unet_c1
    n_b_out_unet_c1 = __icse_30
    n_cat2pad_unet_c1 = c32_unet_c1 * hp2_unet_c1 * wp2_unet_c1
    n_d2_mid_unet_c1 = __icse_28
    n_d2_midpad_unet_c1 = __icse_29
    n_d2_out_unet_c1 = __icse_28
    n_cat1pad_unet_c1 = c21_unet_c1 * hp1_unet_c1 * wp1_unet_c1
    n_d1_mid_unet_c1 = __icse_26
    n_d1_midpad_unet_c1 = __icse_27
    n_d1_out_unet_c1 = __icse_26
    if div(1 - c_out * h * w, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c_out * h * w, -1) + 1 jacc_kernel_unet_loss_b_52!(c_out, h, lossb, target, targetb, w, y, yb)
    end
    if div(1 - c_out * hw_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c_out * hw_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_53!(b_outb, c1, c_out, dec1out, dec1outb, hw_unet_c1, kh_out_unet_c1, khkw_out_unet_c1, kw_out_unet_c1, w, w_out, w_outb, yb)
    end
    if div(1 - n_d1_out_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_d1_out_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_54!(dec1out, dec1out_stack, dec1outb, n_d1_out_unet_c1, zero_val_unet_c1)
    end
    if div(1 - c1 * hw_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c1 * hw_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_55!(b_d1bb, c1, dec1outb, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d1pad, t_d1padb, w, w_d1b, w_d1bb, wp1_unet_c1)
    end
    if div(1 - c1 * hw_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c1 * hw_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_56!(c1, hp1_unet_c1, hw_unet_c1, pad_unet_c1, t_d1b, t_d1padb, w, wp1_unet_c1)
    end
    if div(1 - n_d1_midpad_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_d1_midpad_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_57!(n_d1_midpad_unet_c1, t_d1padb)
    end
    if div(1 - n_d1_mid_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_d1_mid_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_58!(n_d1_mid_unet_c1, t_d1, t_d1_stack, t_d1b, zero_val_unet_c1)
    end
    if div(1 - c1 * hw_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c1 * hw_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_59!(b_d1ab, c1, c21_unet_c1, cat1pad, cat1padb, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d1b, w, w_d1a, w_d1ab, wp1_unet_c1)
    end
    if div(1 - c21_unet_c1 * hw_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c21_unet_c1 * hw_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_60!(c21_unet_c1, cat1b, cat1padb, hp1_unet_c1, hw_unet_c1, pad_unet_c1, w, wp1_unet_c1)
    end
    if div(1 - n_cat1pad_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_cat1pad_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_61!(cat1padb, n_cat1pad_unet_c1)
    end
    if div(1 - c1 * hw_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c1 * hw_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_62!(c1, c2, cat1b, hw_unet_c1, skip1b)
    end
    if div(1 - c2 * hw_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c2 * hw_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_63!(c2, cat1b, hw_unet_c1, u1b)
    end
    if div(1 - c2 * hw_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c2 * hw_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_64!(c2, dec2outb, hw2_unet_c1, hw_unet_c1, scale_unet_c1, u1b, w, w2_unet_c1)
    end
    if div(1 - n_d2_out_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_d2_out_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_65!(dec2out, dec2out_stack, dec2outb, n_d2_out_unet_c1, zero_val_unet_c1)
    end
    if div(1 - c2 * hw2_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c2 * hw2_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_66!(b_d2bb, c2, dec2outb, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d2pad, t_d2padb, w2_unet_c1, w_d2b, w_d2bb, wp2_unet_c1)
    end
    if div(1 - c2 * hw2_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c2 * hw2_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_67!(c2, hp2_unet_c1, hw2_unet_c1, pad_unet_c1, t_d2b, t_d2padb, w2_unet_c1, wp2_unet_c1)
    end
    if div(1 - n_d2_midpad_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_d2_midpad_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_68!(n_d2_midpad_unet_c1, t_d2padb)
    end
    if div(1 - n_d2_mid_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_d2_mid_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_69!(n_d2_mid_unet_c1, t_d2, t_d2_stack, t_d2b, zero_val_unet_c1)
    end
    if div(1 - c2 * hw2_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c2 * hw2_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_70!(b_d2ab, c2, c32_unet_c1, cat2pad, cat2padb, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d2b, w2_unet_c1, w_d2a, w_d2ab, wp2_unet_c1)
    end
    if div(1 - c32_unet_c1 * hw2_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c32_unet_c1 * hw2_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_71!(c32_unet_c1, cat2b, cat2padb, hp2_unet_c1, hw2_unet_c1, pad_unet_c1, w2_unet_c1, wp2_unet_c1)
    end
    if div(1 - n_cat2pad_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_cat2pad_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_72!(cat2padb, n_cat2pad_unet_c1)
    end
    if div(1 - c2 * hw2_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c2 * hw2_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_73!(c2, c3, cat2b, hw2_unet_c1, skip2b)
    end
    if div(1 - c3 * hw2_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c3 * hw2_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_74!(c3, cat2b, hw2_unet_c1, u2b)
    end
    if div(1 - c3 * hw2_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c3 * hw2_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_75!(bottb, c3, hw2_unet_c1, hw4_unet_c1, scale_unet_c1, u2b, w2_unet_c1, w4_unet_c1)
    end
    if div(1 - n_b_out_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_b_out_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_76!(bott, bott_stack, bottb, n_b_out_unet_c1, zero_val_unet_c1)
    end
    if div(1 - c3 * hw4_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c3 * hw4_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_77!(b_bbb, bottb, c3, hp4_unet_c1, hw4_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_bpad, t_bpadb, w4_unet_c1, w_bb, w_bbb, wp4_unet_c1)
    end
    if div(1 - c3 * hw4_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c3 * hw4_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_78!(c3, hp4_unet_c1, hw4_unet_c1, pad_unet_c1, t_bb, t_bpadb, w4_unet_c1, wp4_unet_c1)
    end
    if div(1 - n_b_midpad_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_b_midpad_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_79!(n_b_midpad_unet_c1, t_bpadb)
    end
    if div(1 - n_b_mid_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_b_mid_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_80!(n_b_mid_unet_c1, t_b, t_b_stack, t_bb, zero_val_unet_c1)
    end
    if div(1 - c3 * hw4_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c3 * hw4_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_81!(b_bab, c2, c3, hp4_unet_c1, hw4_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, p2pad, p2padb, t_bb, w4_unet_c1, w_ba, w_bab, wp4_unet_c1)
    end
    if div(1 - c2 * hw4_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c2 * hw4_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_82!(c2, hp4_unet_c1, hw4_unet_c1, p2b, p2padb, pad_unet_c1, w4_unet_c1, wp4_unet_c1)
    end
    if div(1 - n_p2pad_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_p2pad_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_83!(n_p2pad_unet_c1, p2padb)
    end
    if div(c2 * hw4_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw4_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_84!(c2, hw2_unet_c1, hw4_unet_c1, p2b, skip2, skip2b, w2_unet_c1, w4_unet_c1)
    end
    if div(1 - n_e2_out_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_e2_out_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_85!(n_e2_out_unet_c1, skip2, skip2_stack, skip2b, zero_val_unet_c1)
    end
    if div(1 - c2 * hw2_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c2 * hw2_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_86!(b_e2bb, c2, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, skip2b, t_e2pad, t_e2padb, w2_unet_c1, w_e2b, w_e2bb, wp2_unet_c1)
    end
    if div(1 - c2 * hw2_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c2 * hw2_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_87!(c2, hp2_unet_c1, hw2_unet_c1, pad_unet_c1, t_e2b, t_e2padb, w2_unet_c1, wp2_unet_c1)
    end
    if div(1 - n_e2_midpad_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_e2_midpad_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_88!(n_e2_midpad_unet_c1, t_e2padb)
    end
    if div(1 - n_e2_mid_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_e2_mid_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_89!(n_e2_mid_unet_c1, t_e2, t_e2_stack, t_e2b, zero_val_unet_c1)
    end
    if div(1 - c2 * hw2_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c2 * hw2_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_90!(b_e2ab, c1, c2, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, p1pad, p1padb, t_e2b, w2_unet_c1, w_e2a, w_e2ab, wp2_unet_c1)
    end
    if div(1 - c1 * hw2_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c1 * hw2_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_91!(c1, hp2_unet_c1, hw2_unet_c1, p1b, p1padb, pad_unet_c1, w2_unet_c1, wp2_unet_c1)
    end
    if div(1 - n_p1pad_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_p1pad_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_92!(n_p1pad_unet_c1, p1padb)
    end
    if div(c1 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_b_93!(c1, hw2_unet_c1, hw_unet_c1, p1b, skip1, skip1b, w, w2_unet_c1)
    end
    if div(1 - n_e1_out_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_e1_out_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_94!(n_e1_out_unet_c1, skip1, skip1_stack, skip1b, zero_val_unet_c1)
    end
    if div(1 - c1 * hw_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c1 * hw_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_95!(b_e1bb, c1, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, skip1b, t_e1pad, t_e1padb, w, w_e1b, w_e1bb, wp1_unet_c1)
    end
    if div(1 - c1 * hw_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c1 * hw_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_96!(c1, hp1_unet_c1, hw_unet_c1, pad_unet_c1, t_e1b, t_e1padb, w, wp1_unet_c1)
    end
    if div(1 - n_e1_midpad_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_e1_midpad_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_97!(n_e1_midpad_unet_c1, t_e1padb)
    end
    if div(1 - n_e1_mid_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_e1_mid_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_98!(n_e1_mid_unet_c1, t_e1, t_e1_stack, t_e1b, zero_val_unet_c1)
    end
    if div(1 - c1 * hw_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c1 * hw_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_99!(b_e1ab, c1, c_in, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_e1b, w, w_e1a, w_e1ab, wp1_unet_c1, xpad0, xpad0b)
    end
    if div(1 - c_in * hw_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - c_in * hw_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_100!(c_in, hp1_unet_c1, hw_unet_c1, pad_unet_c1, w, wp1_unet_c1, xb, xpad0b)
    end
    if div(1 - n_xpad0_unet_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_xpad0_unet_c1, -1) + 1 jacc_kernel_unet_loss_b_101!(n_xpad0_unet_c1, xpad0b)
    end
    zero_val_unet_c1b = 0.0
    return nothing
end

function unet_loss_jacc(x, h, w, c_in, c1, c2, c3, c_out, w_e1a, b_e1a, w_e1b, b_e1b, w_e2a, b_e2a, w_e2b, b_e2b, w_ba, b_ba, w_bb, b_bb, w_d2a, b_d2a, w_d2b, b_d2b, w_d1a, b_d1a, w_d1b, b_d1b, w_out, b_out, xpad0, t_e1, t_e1pad, skip1, p1, p1pad, t_e2, t_e2pad, skip2, p2, p2pad, t_b, t_bpad, bott, u2, cat2, cat2pad, t_d2, t_d2pad, dec2out, u1, cat1, cat1pad, t_d1, t_d1pad, dec1out, y, target, loss)
    two_unet_c1 = 2
    four_unet_c1 = 4
    h2_unet_c1 = div(h, two_unet_c1)
    w2_unet_c1 = div(w, two_unet_c1)
    h4_unet_c1 = div(h, four_unet_c1)
    w4_unet_c1 = div(w, four_unet_c1)
    hp1_unet_c1 = h + 2
    wp1_unet_c1 = w + 2
    hp2_unet_c1 = h2_unet_c1 + 2
    wp2_unet_c1 = w2_unet_c1 + 2
    hp4_unet_c1 = h4_unet_c1 + 2
    wp4_unet_c1 = w4_unet_c1 + 2
    pad_unet_c1 = 1
    kh_unet_c1 = 3
    kw_unet_c1 = 3
    khkw_unet_c1 = kh_unet_c1 * kw_unet_c1
    kh_out_unet_c1 = 1
    kw_out_unet_c1 = 1
    khkw_out_unet_c1 = kh_out_unet_c1 * kw_out_unet_c1
    scale_unet_c1 = 2
    zero_val_unet_c1 = 0.0
    c32_unet_c1 = c3 + c2
    c21_unet_c1 = c2 + c1
    hw_unet_c1 = h * w
    hw2_unet_c1 = h2_unet_c1 * w2_unet_c1
    hw4_unet_c1 = h4_unet_c1 * w4_unet_c1
    n_xpad0_unet_c1 = c_in * hp1_unet_c1 * wp1_unet_c1
    n_e1_mid_unet_c1 = c1 * hw_unet_c1
    n_e1_midpad_unet_c1 = c1 * hp1_unet_c1 * wp1_unet_c1
    n_e1_out_unet_c1 = c1 * hw_unet_c1
    n_p1pad_unet_c1 = c1 * hp2_unet_c1 * wp2_unet_c1
    n_e2_mid_unet_c1 = c2 * hw2_unet_c1
    n_e2_midpad_unet_c1 = c2 * hp2_unet_c1 * wp2_unet_c1
    n_e2_out_unet_c1 = c2 * hw2_unet_c1
    n_p2pad_unet_c1 = c2 * hp4_unet_c1 * wp4_unet_c1
    n_b_mid_unet_c1 = c3 * hw4_unet_c1
    n_b_midpad_unet_c1 = c3 * hp4_unet_c1 * wp4_unet_c1
    n_b_out_unet_c1 = c3 * hw4_unet_c1
    n_cat2pad_unet_c1 = c32_unet_c1 * hp2_unet_c1 * wp2_unet_c1
    n_d2_mid_unet_c1 = c2 * hw2_unet_c1
    n_d2_midpad_unet_c1 = c2 * hp2_unet_c1 * wp2_unet_c1
    n_d2_out_unet_c1 = c2 * hw2_unet_c1
    n_cat1pad_unet_c1 = c21_unet_c1 * hp1_unet_c1 * wp1_unet_c1
    n_d1_mid_unet_c1 = c1 * hw_unet_c1
    n_d1_midpad_unet_c1 = c1 * hp1_unet_c1 * wp1_unet_c1
    n_d1_out_unet_c1 = c1 * hw_unet_c1
    if div(n_xpad0_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_xpad0_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_1!(n_xpad0_unet_c1, xpad0, zero_val_unet_c1)
    end
    if div(c_in * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c_in * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_2!(c_in, hp1_unet_c1, hw_unet_c1, pad_unet_c1, w, wp1_unet_c1, x, xpad0)
    end
    if div(c1 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_3!(b_e1a, c1, c_in, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_e1, w, w_e1a, wp1_unet_c1, xpad0)
    end
    if div(n_e1_mid_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_e1_mid_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_4!(n_e1_mid_unet_c1, t_e1, zero_val_unet_c1)
    end
    if div(n_e1_midpad_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_e1_midpad_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_5!(n_e1_midpad_unet_c1, t_e1pad, zero_val_unet_c1)
    end
    if div(c1 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_6!(c1, hp1_unet_c1, hw_unet_c1, pad_unet_c1, t_e1, t_e1pad, w, wp1_unet_c1)
    end
    if div(c1 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_7!(b_e1b, c1, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, skip1, t_e1pad, w, w_e1b, wp1_unet_c1)
    end
    if div(n_e1_out_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_e1_out_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_8!(n_e1_out_unet_c1, skip1, zero_val_unet_c1)
    end
    if div(c1 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_9!(c1, hw2_unet_c1, hw_unet_c1, p1, skip1, w, w2_unet_c1)
    end
    if div(n_p1pad_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_p1pad_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_10!(n_p1pad_unet_c1, p1pad, zero_val_unet_c1)
    end
    if div(c1 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_11!(c1, hp2_unet_c1, hw2_unet_c1, p1, p1pad, pad_unet_c1, w2_unet_c1, wp2_unet_c1)
    end
    if div(c2 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_12!(b_e2a, c1, c2, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, p1pad, t_e2, w2_unet_c1, w_e2a, wp2_unet_c1)
    end
    if div(n_e2_mid_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_e2_mid_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_13!(n_e2_mid_unet_c1, t_e2, zero_val_unet_c1)
    end
    if div(n_e2_midpad_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_e2_midpad_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_14!(n_e2_midpad_unet_c1, t_e2pad, zero_val_unet_c1)
    end
    if div(c2 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_15!(c2, hp2_unet_c1, hw2_unet_c1, pad_unet_c1, t_e2, t_e2pad, w2_unet_c1, wp2_unet_c1)
    end
    if div(c2 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_16!(b_e2b, c2, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, skip2, t_e2pad, w2_unet_c1, w_e2b, wp2_unet_c1)
    end
    if div(n_e2_out_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_e2_out_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_17!(n_e2_out_unet_c1, skip2, zero_val_unet_c1)
    end
    if div(c2 * hw4_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw4_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_18!(c2, hw2_unet_c1, hw4_unet_c1, p2, skip2, w2_unet_c1, w4_unet_c1)
    end
    if div(n_p2pad_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_p2pad_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_19!(n_p2pad_unet_c1, p2pad, zero_val_unet_c1)
    end
    if div(c2 * hw4_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw4_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_20!(c2, hp4_unet_c1, hw4_unet_c1, p2, p2pad, pad_unet_c1, w4_unet_c1, wp4_unet_c1)
    end
    if div(c3 * hw4_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c3 * hw4_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_21!(b_ba, c2, c3, hp4_unet_c1, hw4_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, p2pad, t_b, w4_unet_c1, w_ba, wp4_unet_c1)
    end
    if div(n_b_mid_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_b_mid_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_22!(n_b_mid_unet_c1, t_b, zero_val_unet_c1)
    end
    if div(n_b_midpad_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_b_midpad_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_23!(n_b_midpad_unet_c1, t_bpad, zero_val_unet_c1)
    end
    if div(c3 * hw4_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c3 * hw4_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_24!(c3, hp4_unet_c1, hw4_unet_c1, pad_unet_c1, t_b, t_bpad, w4_unet_c1, wp4_unet_c1)
    end
    if div(c3 * hw4_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c3 * hw4_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_25!(b_bb, bott, c3, hp4_unet_c1, hw4_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_bpad, w4_unet_c1, w_bb, wp4_unet_c1)
    end
    if div(n_b_out_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_b_out_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_26!(bott, n_b_out_unet_c1, zero_val_unet_c1)
    end
    if div(c3 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c3 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_27!(bott, c3, hw2_unet_c1, hw4_unet_c1, scale_unet_c1, u2, w2_unet_c1, w4_unet_c1)
    end
    if div(c3 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c3 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_28!(c3, cat2, hw2_unet_c1, u2)
    end
    if div(c2 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_29!(c2, c3, cat2, hw2_unet_c1, skip2)
    end
    if div(n_cat2pad_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_cat2pad_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_30!(cat2pad, n_cat2pad_unet_c1, zero_val_unet_c1)
    end
    if div(c32_unet_c1 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c32_unet_c1 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_31!(c32_unet_c1, cat2, cat2pad, hp2_unet_c1, hw2_unet_c1, pad_unet_c1, w2_unet_c1, wp2_unet_c1)
    end
    if div(c2 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_32!(b_d2a, c2, c32_unet_c1, cat2pad, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d2, w2_unet_c1, w_d2a, wp2_unet_c1)
    end
    if div(n_d2_mid_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_d2_mid_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_33!(n_d2_mid_unet_c1, t_d2, zero_val_unet_c1)
    end
    if div(n_d2_midpad_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_d2_midpad_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_34!(n_d2_midpad_unet_c1, t_d2pad, zero_val_unet_c1)
    end
    if div(c2 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_35!(c2, hp2_unet_c1, hw2_unet_c1, pad_unet_c1, t_d2, t_d2pad, w2_unet_c1, wp2_unet_c1)
    end
    if div(c2 * hw2_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_36!(b_d2b, c2, dec2out, hp2_unet_c1, hw2_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d2pad, w2_unet_c1, w_d2b, wp2_unet_c1)
    end
    if div(n_d2_out_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_d2_out_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_37!(dec2out, n_d2_out_unet_c1, zero_val_unet_c1)
    end
    if div(c2 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_38!(c2, dec2out, hw2_unet_c1, hw_unet_c1, scale_unet_c1, u1, w, w2_unet_c1)
    end
    if div(c2 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_39!(c2, cat1, hw_unet_c1, u1)
    end
    if div(c1 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_40!(c1, c2, cat1, hw_unet_c1, skip1)
    end
    if div(n_cat1pad_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_cat1pad_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_41!(cat1pad, n_cat1pad_unet_c1, zero_val_unet_c1)
    end
    if div(c21_unet_c1 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c21_unet_c1 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_42!(c21_unet_c1, cat1, cat1pad, hp1_unet_c1, hw_unet_c1, pad_unet_c1, w, wp1_unet_c1)
    end
    if div(c1 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_43!(b_d1a, c1, c21_unet_c1, cat1pad, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d1, w, w_d1a, wp1_unet_c1)
    end
    if div(n_d1_mid_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_d1_mid_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_44!(n_d1_mid_unet_c1, t_d1, zero_val_unet_c1)
    end
    if div(n_d1_midpad_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_d1_midpad_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_45!(n_d1_midpad_unet_c1, t_d1pad, zero_val_unet_c1)
    end
    if div(c1 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_46!(c1, hp1_unet_c1, hw_unet_c1, pad_unet_c1, t_d1, t_d1pad, w, wp1_unet_c1)
    end
    if div(c1 * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_47!(b_d1b, c1, dec1out, hp1_unet_c1, hw_unet_c1, kh_unet_c1, khkw_unet_c1, kw_unet_c1, t_d1pad, w, w_d1b, wp1_unet_c1)
    end
    if div(n_d1_out_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_d1_out_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_48!(dec1out, n_d1_out_unet_c1, zero_val_unet_c1)
    end
    if div(c_out * hw_unet_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c_out * hw_unet_c1 - 1, 1) + 1 jacc_kernel_unet_loss_49!(b_out, c1, c_out, dec1out, hw_unet_c1, kh_out_unet_c1, khkw_out_unet_c1, kw_out_unet_c1, w, w_out, y)
    end
    if div(c_out * h * w - 1, 1) + 1 < 32768
        if div(c_out * h * w - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(c_out * h * w - 1, 1) + 1 jacc_kernel_unet_loss_51!(c_out, h, loss, target, w, y)
        end
    else
        if div(c_out * h * w - 1, 1) + 1 > 0
            __jgen_redval_50 = JACC.@parallel_reduce(range = div(c_out * h * w - 1, 1) + 1, (((i_o, target, y)->(y[i_o] - target[i_o]) ^ 2))(target, y))
            JACC.@parallel_for range = 1 jacc_kernel_unet_loss_50!(loss, __jgen_redval_50)
        end
    end
    return nothing
end
