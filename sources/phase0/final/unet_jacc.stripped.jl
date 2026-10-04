using CUDA
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_unet_1!(__jacc_i, n_xpad0, xpad0, zero_val)
    i = 1 + (__jacc_i - 1)
    xpad0[i] = zero_val
    return nothing
end

function jacc_kernel_unet_2!(__jacc_i, c_in, hp1, hw, pad, w, wp1, x, xpad0)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw) + 1
    rem = mod(idxm1, hw)
    i = div(rem, w) + 1
    j = mod(rem, w) + 1
    yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
    xpad0[yi] = x[idx]
    return nothing
end

function jacc_kernel_unet_3!(__jacc_i, b_e1a, c1, c_in, hp1, hw, kh, khkw, kw, t_e1, w, w_e1a, wp1, xpad0)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    co = div(idxm1, hw) + 1
    rem = mod(idxm1, hw)
    i = div(rem, w) + 1
    j = mod(rem, w) + 1
    s = 0.0
    for i_k = 1:c_in * khkw
        kseqm1 = i_k - 1
        ci = div(kseqm1, khkw) + 1
        rem2 = mod(kseqm1, khkw)
        ki = div(rem2, kw) + 1
        kj = mod(rem2, kw) + 1
        row = (i + ki) - 1
        col = (j + kj) - 1
        xi = (ci - 1) * hp1 * wp1 + (row - 1) * wp1 + col
        wi = (((co - 1) * c_in + (ci - 1)) * kh + (ki - 1)) * kw + kj
        s = s + xpad0[xi] * w_e1a[wi]
    end
    t_e1[idx] = s + b_e1a[co]
    return nothing
end

function jacc_kernel_unet_4!(__jacc_i, n_e1_mid, t_e1, zero_val)
    i = 1 + (__jacc_i - 1)
    t_e1[i] = max(t_e1[i], zero_val)
    return nothing
end

function jacc_kernel_unet_5!(__jacc_i, n_e1_midpad, t_e1pad, zero_val)
    i = 1 + (__jacc_i - 1)
    t_e1pad[i] = zero_val
    return nothing
end

function jacc_kernel_unet_6!(__jacc_i, c1, hp1, hw, pad, t_e1, t_e1pad, w, wp1)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw) + 1
    rem = mod(idxm1, hw)
    i = div(rem, w) + 1
    j = mod(rem, w) + 1
    yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
    t_e1pad[yi] = t_e1[idx]
    return nothing
end

function jacc_kernel_unet_7!(__jacc_i, b_e1b, c1, hp1, hw, kh, khkw, kw, skip1, t_e1pad, w, w_e1b, wp1)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    co = div(idxm1, hw) + 1
    rem = mod(idxm1, hw)
    i = div(rem, w) + 1
    j = mod(rem, w) + 1
    s = 0.0
    for i_k = 1:c1 * khkw
        kseqm1 = i_k - 1
        ci = div(kseqm1, khkw) + 1
        rem2 = mod(kseqm1, khkw)
        ki = div(rem2, kw) + 1
        kj = mod(rem2, kw) + 1
        row = (i + ki) - 1
        col = (j + kj) - 1
        xi = (ci - 1) * hp1 * wp1 + (row - 1) * wp1 + col
        wi = (((co - 1) * c1 + (ci - 1)) * kh + (ki - 1)) * kw + kj
        s = s + t_e1pad[xi] * w_e1b[wi]
    end
    skip1[idx] = s + b_e1b[co]
    return nothing
end

function jacc_kernel_unet_8!(__jacc_i, n_e1_out, skip1, zero_val)
    i = 1 + (__jacc_i - 1)
    skip1[i] = max(skip1[i], zero_val)
    return nothing
end

function jacc_kernel_unet_9!(__jacc_i, c1, hw, hw2, p1, skip1, w, w2)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw2) + 1
    rem = mod(idxm1, hw2)
    oi = div(rem, w2) + 1
    oj = mod(rem, w2) + 1
    i0 = 2oi - 1
    j0 = 2oj - 1
    a11 = skip1[(ci - 1) * hw + (i0 - 1) * w + j0]
    a12 = skip1[(ci - 1) * hw + (i0 - 1) * w + j0 + 1]
    a21 = skip1[(ci - 1) * hw + i0 * w + j0]
    a22 = skip1[(ci - 1) * hw + i0 * w + j0 + 1]
    m1 = max(a11, a12)
    m2 = max(a21, a22)
    p1[idx] = max(m1, m2)
    return nothing
end

function jacc_kernel_unet_10!(__jacc_i, n_p1pad, p1pad, zero_val)
    i = 1 + (__jacc_i - 1)
    p1pad[i] = zero_val
    return nothing
end

function jacc_kernel_unet_11!(__jacc_i, c1, hp2, hw2, p1, p1pad, pad, w2, wp2)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw2) + 1
    rem = mod(idxm1, hw2)
    i = div(rem, w2) + 1
    j = mod(rem, w2) + 1
    yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
    p1pad[yi] = p1[idx]
    return nothing
end

function jacc_kernel_unet_12!(__jacc_i, b_e2a, c1, c2, hp2, hw2, kh, khkw, kw, p1pad, t_e2, w2, w_e2a, wp2)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    co = div(idxm1, hw2) + 1
    rem = mod(idxm1, hw2)
    i = div(rem, w2) + 1
    j = mod(rem, w2) + 1
    s = 0.0
    for i_k = 1:c1 * khkw
        kseqm1 = i_k - 1
        ci = div(kseqm1, khkw) + 1
        rem2 = mod(kseqm1, khkw)
        ki = div(rem2, kw) + 1
        kj = mod(rem2, kw) + 1
        row = (i + ki) - 1
        col = (j + kj) - 1
        xi = (ci - 1) * hp2 * wp2 + (row - 1) * wp2 + col
        wi = (((co - 1) * c1 + (ci - 1)) * kh + (ki - 1)) * kw + kj
        s = s + p1pad[xi] * w_e2a[wi]
    end
    t_e2[idx] = s + b_e2a[co]
    return nothing
end

function jacc_kernel_unet_13!(__jacc_i, n_e2_mid, t_e2, zero_val)
    i = 1 + (__jacc_i - 1)
    t_e2[i] = max(t_e2[i], zero_val)
    return nothing
end

function jacc_kernel_unet_14!(__jacc_i, n_e2_midpad, t_e2pad, zero_val)
    i = 1 + (__jacc_i - 1)
    t_e2pad[i] = zero_val
    return nothing
end

function jacc_kernel_unet_15!(__jacc_i, c2, hp2, hw2, pad, t_e2, t_e2pad, w2, wp2)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw2) + 1
    rem = mod(idxm1, hw2)
    i = div(rem, w2) + 1
    j = mod(rem, w2) + 1
    yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
    t_e2pad[yi] = t_e2[idx]
    return nothing
end

function jacc_kernel_unet_16!(__jacc_i, b_e2b, c2, hp2, hw2, kh, khkw, kw, skip2, t_e2pad, w2, w_e2b, wp2)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    co = div(idxm1, hw2) + 1
    rem = mod(idxm1, hw2)
    i = div(rem, w2) + 1
    j = mod(rem, w2) + 1
    s = 0.0
    for i_k = 1:c2 * khkw
        kseqm1 = i_k - 1
        ci = div(kseqm1, khkw) + 1
        rem2 = mod(kseqm1, khkw)
        ki = div(rem2, kw) + 1
        kj = mod(rem2, kw) + 1
        row = (i + ki) - 1
        col = (j + kj) - 1
        xi = (ci - 1) * hp2 * wp2 + (row - 1) * wp2 + col
        wi = (((co - 1) * c2 + (ci - 1)) * kh + (ki - 1)) * kw + kj
        s = s + t_e2pad[xi] * w_e2b[wi]
    end
    skip2[idx] = s + b_e2b[co]
    return nothing
end

function jacc_kernel_unet_17!(__jacc_i, n_e2_out, skip2, zero_val)
    i = 1 + (__jacc_i - 1)
    skip2[i] = max(skip2[i], zero_val)
    return nothing
end

function jacc_kernel_unet_18!(__jacc_i, c2, hw2, hw4, p2, skip2, w2, w4)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw4) + 1
    rem = mod(idxm1, hw4)
    oi = div(rem, w4) + 1
    oj = mod(rem, w4) + 1
    i0 = 2oi - 1
    j0 = 2oj - 1
    a11 = skip2[(ci - 1) * hw2 + (i0 - 1) * w2 + j0]
    a12 = skip2[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1]
    a21 = skip2[(ci - 1) * hw2 + i0 * w2 + j0]
    a22 = skip2[(ci - 1) * hw2 + i0 * w2 + j0 + 1]
    m1 = max(a11, a12)
    m2 = max(a21, a22)
    p2[idx] = max(m1, m2)
    return nothing
end

function jacc_kernel_unet_19!(__jacc_i, n_p2pad, p2pad, zero_val)
    i = 1 + (__jacc_i - 1)
    p2pad[i] = zero_val
    return nothing
end

function jacc_kernel_unet_20!(__jacc_i, c2, hp4, hw4, p2, p2pad, pad, w4, wp4)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw4) + 1
    rem = mod(idxm1, hw4)
    i = div(rem, w4) + 1
    j = mod(rem, w4) + 1
    yi = (ci - 1) * hp4 * wp4 + ((i + pad) - 1) * wp4 + (j + pad)
    p2pad[yi] = p2[idx]
    return nothing
end

function jacc_kernel_unet_21!(__jacc_i, b_ba, c2, c3, hp4, hw4, kh, khkw, kw, p2pad, t_b, w4, w_ba, wp4)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    co = div(idxm1, hw4) + 1
    rem = mod(idxm1, hw4)
    i = div(rem, w4) + 1
    j = mod(rem, w4) + 1
    s = 0.0
    for i_k = 1:c2 * khkw
        kseqm1 = i_k - 1
        ci = div(kseqm1, khkw) + 1
        rem2 = mod(kseqm1, khkw)
        ki = div(rem2, kw) + 1
        kj = mod(rem2, kw) + 1
        row = (i + ki) - 1
        col = (j + kj) - 1
        xi = (ci - 1) * hp4 * wp4 + (row - 1) * wp4 + col
        wi = (((co - 1) * c2 + (ci - 1)) * kh + (ki - 1)) * kw + kj
        s = s + p2pad[xi] * w_ba[wi]
    end
    t_b[idx] = s + b_ba[co]
    return nothing
end

function jacc_kernel_unet_22!(__jacc_i, n_b_mid, t_b, zero_val)
    i = 1 + (__jacc_i - 1)
    t_b[i] = max(t_b[i], zero_val)
    return nothing
end

function jacc_kernel_unet_23!(__jacc_i, n_b_midpad, t_bpad, zero_val)
    i = 1 + (__jacc_i - 1)
    t_bpad[i] = zero_val
    return nothing
end

function jacc_kernel_unet_24!(__jacc_i, c3, hp4, hw4, pad, t_b, t_bpad, w4, wp4)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw4) + 1
    rem = mod(idxm1, hw4)
    i = div(rem, w4) + 1
    j = mod(rem, w4) + 1
    yi = (ci - 1) * hp4 * wp4 + ((i + pad) - 1) * wp4 + (j + pad)
    t_bpad[yi] = t_b[idx]
    return nothing
end

function jacc_kernel_unet_25!(__jacc_i, b_bb, bott, c3, hp4, hw4, kh, khkw, kw, t_bpad, w4, w_bb, wp4)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    co = div(idxm1, hw4) + 1
    rem = mod(idxm1, hw4)
    i = div(rem, w4) + 1
    j = mod(rem, w4) + 1
    s = 0.0
    for i_k = 1:c3 * khkw
        kseqm1 = i_k - 1
        ci = div(kseqm1, khkw) + 1
        rem2 = mod(kseqm1, khkw)
        ki = div(rem2, kw) + 1
        kj = mod(rem2, kw) + 1
        row = (i + ki) - 1
        col = (j + kj) - 1
        xi = (ci - 1) * hp4 * wp4 + (row - 1) * wp4 + col
        wi = (((co - 1) * c3 + (ci - 1)) * kh + (ki - 1)) * kw + kj
        s = s + t_bpad[xi] * w_bb[wi]
    end
    bott[idx] = s + b_bb[co]
    return nothing
end

function jacc_kernel_unet_26!(__jacc_i, bott, n_b_out, zero_val)
    i = 1 + (__jacc_i - 1)
    bott[i] = max(bott[i], zero_val)
    return nothing
end

function jacc_kernel_unet_27!(__jacc_i, bott, c3, hw2, hw4, scale, u2, w2, w4)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw2) + 1
    rem = mod(idxm1, hw2)
    oi = div(rem, w2) + 1
    oj = mod(rem, w2) + 1
    oim1 = oi - 1
    ojm1 = oj - 1
    i = div(oim1, scale) + 1
    j = div(ojm1, scale) + 1
    xi = (ci - 1) * hw4 + (i - 1) * w4 + j
    u2[idx] = bott[xi]
    return nothing
end

function jacc_kernel_unet_28!(__jacc_i, c3, cat2, hw2, u2)
    idx = 1 + (__jacc_i - 1)
    cat2[idx] = u2[idx]
    return nothing
end

function jacc_kernel_unet_29!(__jacc_i, c2, c3, cat2, hw2, skip2)
    idx = 1 + (__jacc_i - 1)
    cat2[c3 * hw2 + idx] = skip2[idx]
    return nothing
end

function jacc_kernel_unet_30!(__jacc_i, cat2pad, n_cat2pad, zero_val)
    i = 1 + (__jacc_i - 1)
    cat2pad[i] = zero_val
    return nothing
end

function jacc_kernel_unet_31!(__jacc_i, c32, cat2, cat2pad, hp2, hw2, pad, w2, wp2)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw2) + 1
    rem = mod(idxm1, hw2)
    i = div(rem, w2) + 1
    j = mod(rem, w2) + 1
    yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
    cat2pad[yi] = cat2[idx]
    return nothing
end

function jacc_kernel_unet_32!(__jacc_i, b_d2a, c2, c32, cat2pad, hp2, hw2, kh, khkw, kw, t_d2, w2, w_d2a, wp2)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    co = div(idxm1, hw2) + 1
    rem = mod(idxm1, hw2)
    i = div(rem, w2) + 1
    j = mod(rem, w2) + 1
    s = 0.0
    for i_k = 1:c32 * khkw
        kseqm1 = i_k - 1
        ci = div(kseqm1, khkw) + 1
        rem2 = mod(kseqm1, khkw)
        ki = div(rem2, kw) + 1
        kj = mod(rem2, kw) + 1
        row = (i + ki) - 1
        col = (j + kj) - 1
        xi = (ci - 1) * hp2 * wp2 + (row - 1) * wp2 + col
        wi = (((co - 1) * c32 + (ci - 1)) * kh + (ki - 1)) * kw + kj
        s = s + cat2pad[xi] * w_d2a[wi]
    end
    t_d2[idx] = s + b_d2a[co]
    return nothing
end

function jacc_kernel_unet_33!(__jacc_i, n_d2_mid, t_d2, zero_val)
    i = 1 + (__jacc_i - 1)
    t_d2[i] = max(t_d2[i], zero_val)
    return nothing
end

function jacc_kernel_unet_34!(__jacc_i, n_d2_midpad, t_d2pad, zero_val)
    i = 1 + (__jacc_i - 1)
    t_d2pad[i] = zero_val
    return nothing
end

function jacc_kernel_unet_35!(__jacc_i, c2, hp2, hw2, pad, t_d2, t_d2pad, w2, wp2)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw2) + 1
    rem = mod(idxm1, hw2)
    i = div(rem, w2) + 1
    j = mod(rem, w2) + 1
    yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
    t_d2pad[yi] = t_d2[idx]
    return nothing
end

function jacc_kernel_unet_36!(__jacc_i, b_d2b, c2, dec2out, hp2, hw2, kh, khkw, kw, t_d2pad, w2, w_d2b, wp2)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    co = div(idxm1, hw2) + 1
    rem = mod(idxm1, hw2)
    i = div(rem, w2) + 1
    j = mod(rem, w2) + 1
    s = 0.0
    for i_k = 1:c2 * khkw
        kseqm1 = i_k - 1
        ci = div(kseqm1, khkw) + 1
        rem2 = mod(kseqm1, khkw)
        ki = div(rem2, kw) + 1
        kj = mod(rem2, kw) + 1
        row = (i + ki) - 1
        col = (j + kj) - 1
        xi = (ci - 1) * hp2 * wp2 + (row - 1) * wp2 + col
        wi = (((co - 1) * c2 + (ci - 1)) * kh + (ki - 1)) * kw + kj
        s = s + t_d2pad[xi] * w_d2b[wi]
    end
    dec2out[idx] = s + b_d2b[co]
    return nothing
end

function jacc_kernel_unet_37!(__jacc_i, dec2out, n_d2_out, zero_val)
    i = 1 + (__jacc_i - 1)
    dec2out[i] = max(dec2out[i], zero_val)
    return nothing
end

function jacc_kernel_unet_38!(__jacc_i, c2, dec2out, hw, hw2, scale, u1, w, w2)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw) + 1
    rem = mod(idxm1, hw)
    oi = div(rem, w) + 1
    oj = mod(rem, w) + 1
    oim1 = oi - 1
    ojm1 = oj - 1
    i = div(oim1, scale) + 1
    j = div(ojm1, scale) + 1
    xi = (ci - 1) * hw2 + (i - 1) * w2 + j
    u1[idx] = dec2out[xi]
    return nothing
end

function jacc_kernel_unet_39!(__jacc_i, c2, cat1, hw, u1)
    idx = 1 + (__jacc_i - 1)
    cat1[idx] = u1[idx]
    return nothing
end

function jacc_kernel_unet_40!(__jacc_i, c1, c2, cat1, hw, skip1)
    idx = 1 + (__jacc_i - 1)
    cat1[c2 * hw + idx] = skip1[idx]
    return nothing
end

function jacc_kernel_unet_41!(__jacc_i, cat1pad, n_cat1pad, zero_val)
    i = 1 + (__jacc_i - 1)
    cat1pad[i] = zero_val
    return nothing
end

function jacc_kernel_unet_42!(__jacc_i, c21, cat1, cat1pad, hp1, hw, pad, w, wp1)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw) + 1
    rem = mod(idxm1, hw)
    i = div(rem, w) + 1
    j = mod(rem, w) + 1
    yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
    cat1pad[yi] = cat1[idx]
    return nothing
end

function jacc_kernel_unet_43!(__jacc_i, b_d1a, c1, c21, cat1pad, hp1, hw, kh, khkw, kw, t_d1, w, w_d1a, wp1)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    co = div(idxm1, hw) + 1
    rem = mod(idxm1, hw)
    i = div(rem, w) + 1
    j = mod(rem, w) + 1
    s = 0.0
    for i_k = 1:c21 * khkw
        kseqm1 = i_k - 1
        ci = div(kseqm1, khkw) + 1
        rem2 = mod(kseqm1, khkw)
        ki = div(rem2, kw) + 1
        kj = mod(rem2, kw) + 1
        row = (i + ki) - 1
        col = (j + kj) - 1
        xi = (ci - 1) * hp1 * wp1 + (row - 1) * wp1 + col
        wi = (((co - 1) * c21 + (ci - 1)) * kh + (ki - 1)) * kw + kj
        s = s + cat1pad[xi] * w_d1a[wi]
    end
    t_d1[idx] = s + b_d1a[co]
    return nothing
end

function jacc_kernel_unet_44!(__jacc_i, n_d1_mid, t_d1, zero_val)
    i = 1 + (__jacc_i - 1)
    t_d1[i] = max(t_d1[i], zero_val)
    return nothing
end

function jacc_kernel_unet_45!(__jacc_i, n_d1_midpad, t_d1pad, zero_val)
    i = 1 + (__jacc_i - 1)
    t_d1pad[i] = zero_val
    return nothing
end

function jacc_kernel_unet_46!(__jacc_i, c1, hp1, hw, pad, t_d1, t_d1pad, w, wp1)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw) + 1
    rem = mod(idxm1, hw)
    i = div(rem, w) + 1
    j = mod(rem, w) + 1
    yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
    t_d1pad[yi] = t_d1[idx]
    return nothing
end

function jacc_kernel_unet_47!(__jacc_i, b_d1b, c1, dec1out, hp1, hw, kh, khkw, kw, t_d1pad, w, w_d1b, wp1)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    co = div(idxm1, hw) + 1
    rem = mod(idxm1, hw)
    i = div(rem, w) + 1
    j = mod(rem, w) + 1
    s = 0.0
    for i_k = 1:c1 * khkw
        kseqm1 = i_k - 1
        ci = div(kseqm1, khkw) + 1
        rem2 = mod(kseqm1, khkw)
        ki = div(rem2, kw) + 1
        kj = mod(rem2, kw) + 1
        row = (i + ki) - 1
        col = (j + kj) - 1
        xi = (ci - 1) * hp1 * wp1 + (row - 1) * wp1 + col
        wi = (((co - 1) * c1 + (ci - 1)) * kh + (ki - 1)) * kw + kj
        s = s + t_d1pad[xi] * w_d1b[wi]
    end
    dec1out[idx] = s + b_d1b[co]
    return nothing
end

function jacc_kernel_unet_48!(__jacc_i, dec1out, n_d1_out, zero_val)
    i = 1 + (__jacc_i - 1)
    dec1out[i] = max(dec1out[i], zero_val)
    return nothing
end

function jacc_kernel_unet_49!(__jacc_i, b_out, c1, c_out, dec1out, hw, kh_out, khkw_out, kw_out, w, w_out, y)
    idx = 1 + (__jacc_i - 1)
    idxm1 = idx - 1
    co = div(idxm1, hw) + 1
    rem = mod(idxm1, hw)
    i = div(rem, w) + 1
    j = mod(rem, w) + 1
    s = 0.0
    for i_k = 1:c1 * khkw_out
        kseqm1 = i_k - 1
        ci = div(kseqm1, khkw_out) + 1
        rem2 = mod(kseqm1, khkw_out)
        ki = div(rem2, kw_out) + 1
        kj = mod(rem2, kw_out) + 1
        row = (i + ki) - 1
        col = (j + kj) - 1
        xi = (ci - 1) * hw + (row - 1) * w + col
        wi = (((co - 1) * c1 + (ci - 1)) * kh_out + (ki - 1)) * kw_out + kj
        s = s + dec1out[xi] * w_out[wi]
    end
    y[idx] = s + b_out[co]
    return nothing
end

function unet_jacc(x, h, w, c_in, c1, c2, c3, c_out, w_e1a, b_e1a, w_e1b, b_e1b, w_e2a, b_e2a, w_e2b, b_e2b, w_ba, b_ba, w_bb, b_bb, w_d2a, b_d2a, w_d2b, b_d2b, w_d1a, b_d1a, w_d1b, b_d1b, w_out, b_out, xpad0, t_e1, t_e1pad, skip1, p1, p1pad, t_e2, t_e2pad, skip2, p2, p2pad, t_b, t_bpad, bott, u2, cat2, cat2pad, t_d2, t_d2pad, dec2out, u1, cat1, cat1pad, t_d1, t_d1pad, dec1out, y)
    two = 2
    four = 4
    h2 = div(h, two)
    w2 = div(w, two)
    h4 = div(h, four)
    w4 = div(w, four)
    hp1 = h + 2
    wp1 = w + 2
    hp2 = h2 + 2
    wp2 = w2 + 2
    hp4 = h4 + 2
    wp4 = w4 + 2
    pad = 1
    kh = 3
    kw = 3
    khkw = kh * kw
    kh_out = 1
    kw_out = 1
    khkw_out = kh_out * kw_out
    scale = 2
    zero_val = 0.0
    c32 = c3 + c2
    c21 = c2 + c1
    hw = h * w
    hw2 = h2 * w2
    hw4 = h4 * w4
    n_xpad0 = c_in * hp1 * wp1
    n_e1_mid = c1 * hw
    n_e1_midpad = c1 * hp1 * wp1
    n_e1_out = c1 * hw
    n_p1pad = c1 * hp2 * wp2
    n_e2_mid = c2 * hw2
    n_e2_midpad = c2 * hp2 * wp2
    n_e2_out = c2 * hw2
    n_p2pad = c2 * hp4 * wp4
    n_b_mid = c3 * hw4
    n_b_midpad = c3 * hp4 * wp4
    n_b_out = c3 * hw4
    n_cat2pad = c32 * hp2 * wp2
    n_d2_mid = c2 * hw2
    n_d2_midpad = c2 * hp2 * wp2
    n_d2_out = c2 * hw2
    n_cat1pad = c21 * hp1 * wp1
    n_d1_mid = c1 * hw
    n_d1_midpad = c1 * hp1 * wp1
    n_d1_out = c1 * hw
    if div(n_xpad0 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_xpad0 - 1, 1) + 1 jacc_kernel_unet_1!(n_xpad0, xpad0, zero_val)
    end
    if div(c_in * hw - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c_in * hw - 1, 1) + 1 jacc_kernel_unet_2!(c_in, hp1, hw, pad, w, wp1, x, xpad0)
    end
    if div(c1 * hw - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw - 1, 1) + 1 jacc_kernel_unet_3!(b_e1a, c1, c_in, hp1, hw, kh, khkw, kw, t_e1, w, w_e1a, wp1, xpad0)
    end
    if div(n_e1_mid - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_e1_mid - 1, 1) + 1 jacc_kernel_unet_4!(n_e1_mid, t_e1, zero_val)
    end
    if div(n_e1_midpad - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_e1_midpad - 1, 1) + 1 jacc_kernel_unet_5!(n_e1_midpad, t_e1pad, zero_val)
    end
    if div(c1 * hw - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw - 1, 1) + 1 jacc_kernel_unet_6!(c1, hp1, hw, pad, t_e1, t_e1pad, w, wp1)
    end
    if div(c1 * hw - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw - 1, 1) + 1 jacc_kernel_unet_7!(b_e1b, c1, hp1, hw, kh, khkw, kw, skip1, t_e1pad, w, w_e1b, wp1)
    end
    if div(n_e1_out - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_e1_out - 1, 1) + 1 jacc_kernel_unet_8!(n_e1_out, skip1, zero_val)
    end
    if div(c1 * hw2 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw2 - 1, 1) + 1 jacc_kernel_unet_9!(c1, hw, hw2, p1, skip1, w, w2)
    end
    if div(n_p1pad - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_p1pad - 1, 1) + 1 jacc_kernel_unet_10!(n_p1pad, p1pad, zero_val)
    end
    if div(c1 * hw2 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw2 - 1, 1) + 1 jacc_kernel_unet_11!(c1, hp2, hw2, p1, p1pad, pad, w2, wp2)
    end
    if div(c2 * hw2 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2 - 1, 1) + 1 jacc_kernel_unet_12!(b_e2a, c1, c2, hp2, hw2, kh, khkw, kw, p1pad, t_e2, w2, w_e2a, wp2)
    end
    if div(n_e2_mid - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_e2_mid - 1, 1) + 1 jacc_kernel_unet_13!(n_e2_mid, t_e2, zero_val)
    end
    if div(n_e2_midpad - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_e2_midpad - 1, 1) + 1 jacc_kernel_unet_14!(n_e2_midpad, t_e2pad, zero_val)
    end
    if div(c2 * hw2 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2 - 1, 1) + 1 jacc_kernel_unet_15!(c2, hp2, hw2, pad, t_e2, t_e2pad, w2, wp2)
    end
    if div(c2 * hw2 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2 - 1, 1) + 1 jacc_kernel_unet_16!(b_e2b, c2, hp2, hw2, kh, khkw, kw, skip2, t_e2pad, w2, w_e2b, wp2)
    end
    if div(n_e2_out - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_e2_out - 1, 1) + 1 jacc_kernel_unet_17!(n_e2_out, skip2, zero_val)
    end
    if div(c2 * hw4 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw4 - 1, 1) + 1 jacc_kernel_unet_18!(c2, hw2, hw4, p2, skip2, w2, w4)
    end
    if div(n_p2pad - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_p2pad - 1, 1) + 1 jacc_kernel_unet_19!(n_p2pad, p2pad, zero_val)
    end
    if div(c2 * hw4 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw4 - 1, 1) + 1 jacc_kernel_unet_20!(c2, hp4, hw4, p2, p2pad, pad, w4, wp4)
    end
    if div(c3 * hw4 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c3 * hw4 - 1, 1) + 1 jacc_kernel_unet_21!(b_ba, c2, c3, hp4, hw4, kh, khkw, kw, p2pad, t_b, w4, w_ba, wp4)
    end
    if div(n_b_mid - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_b_mid - 1, 1) + 1 jacc_kernel_unet_22!(n_b_mid, t_b, zero_val)
    end
    if div(n_b_midpad - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_b_midpad - 1, 1) + 1 jacc_kernel_unet_23!(n_b_midpad, t_bpad, zero_val)
    end
    if div(c3 * hw4 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c3 * hw4 - 1, 1) + 1 jacc_kernel_unet_24!(c3, hp4, hw4, pad, t_b, t_bpad, w4, wp4)
    end
    if div(c3 * hw4 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c3 * hw4 - 1, 1) + 1 jacc_kernel_unet_25!(b_bb, bott, c3, hp4, hw4, kh, khkw, kw, t_bpad, w4, w_bb, wp4)
    end
    if div(n_b_out - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_b_out - 1, 1) + 1 jacc_kernel_unet_26!(bott, n_b_out, zero_val)
    end
    if div(c3 * hw2 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c3 * hw2 - 1, 1) + 1 jacc_kernel_unet_27!(bott, c3, hw2, hw4, scale, u2, w2, w4)
    end
    if div(c3 * hw2 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c3 * hw2 - 1, 1) + 1 jacc_kernel_unet_28!(c3, cat2, hw2, u2)
    end
    if div(c2 * hw2 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2 - 1, 1) + 1 jacc_kernel_unet_29!(c2, c3, cat2, hw2, skip2)
    end
    if div(n_cat2pad - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_cat2pad - 1, 1) + 1 jacc_kernel_unet_30!(cat2pad, n_cat2pad, zero_val)
    end
    if div(c32 * hw2 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c32 * hw2 - 1, 1) + 1 jacc_kernel_unet_31!(c32, cat2, cat2pad, hp2, hw2, pad, w2, wp2)
    end
    if div(c2 * hw2 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2 - 1, 1) + 1 jacc_kernel_unet_32!(b_d2a, c2, c32, cat2pad, hp2, hw2, kh, khkw, kw, t_d2, w2, w_d2a, wp2)
    end
    if div(n_d2_mid - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_d2_mid - 1, 1) + 1 jacc_kernel_unet_33!(n_d2_mid, t_d2, zero_val)
    end
    if div(n_d2_midpad - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_d2_midpad - 1, 1) + 1 jacc_kernel_unet_34!(n_d2_midpad, t_d2pad, zero_val)
    end
    if div(c2 * hw2 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2 - 1, 1) + 1 jacc_kernel_unet_35!(c2, hp2, hw2, pad, t_d2, t_d2pad, w2, wp2)
    end
    if div(c2 * hw2 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw2 - 1, 1) + 1 jacc_kernel_unet_36!(b_d2b, c2, dec2out, hp2, hw2, kh, khkw, kw, t_d2pad, w2, w_d2b, wp2)
    end
    if div(n_d2_out - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_d2_out - 1, 1) + 1 jacc_kernel_unet_37!(dec2out, n_d2_out, zero_val)
    end
    if div(c2 * hw - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw - 1, 1) + 1 jacc_kernel_unet_38!(c2, dec2out, hw, hw2, scale, u1, w, w2)
    end
    if div(c2 * hw - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c2 * hw - 1, 1) + 1 jacc_kernel_unet_39!(c2, cat1, hw, u1)
    end
    if div(c1 * hw - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw - 1, 1) + 1 jacc_kernel_unet_40!(c1, c2, cat1, hw, skip1)
    end
    if div(n_cat1pad - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_cat1pad - 1, 1) + 1 jacc_kernel_unet_41!(cat1pad, n_cat1pad, zero_val)
    end
    if div(c21 * hw - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c21 * hw - 1, 1) + 1 jacc_kernel_unet_42!(c21, cat1, cat1pad, hp1, hw, pad, w, wp1)
    end
    if div(c1 * hw - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw - 1, 1) + 1 jacc_kernel_unet_43!(b_d1a, c1, c21, cat1pad, hp1, hw, kh, khkw, kw, t_d1, w, w_d1a, wp1)
    end
    if div(n_d1_mid - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_d1_mid - 1, 1) + 1 jacc_kernel_unet_44!(n_d1_mid, t_d1, zero_val)
    end
    if div(n_d1_midpad - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_d1_midpad - 1, 1) + 1 jacc_kernel_unet_45!(n_d1_midpad, t_d1pad, zero_val)
    end
    if div(c1 * hw - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw - 1, 1) + 1 jacc_kernel_unet_46!(c1, hp1, hw, pad, t_d1, t_d1pad, w, wp1)
    end
    if div(c1 * hw - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c1 * hw - 1, 1) + 1 jacc_kernel_unet_47!(b_d1b, c1, dec1out, hp1, hw, kh, khkw, kw, t_d1pad, w, w_d1b, wp1)
    end
    if div(n_d1_out - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_d1_out - 1, 1) + 1 jacc_kernel_unet_48!(dec1out, n_d1_out, zero_val)
    end
    if div(c_out * hw - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(c_out * hw - 1, 1) + 1 jacc_kernel_unet_49!(b_out, c1, c_out, dec1out, hw, kh_out, khkw_out, kw_out, w, w_out, y)
    end
    return nothing
end
