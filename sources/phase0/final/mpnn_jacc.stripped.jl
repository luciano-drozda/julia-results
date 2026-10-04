using CUDA
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_mpnn_1!(__jacc_i, agg, n_agg)
    k = 1 + (__jacc_i - 1)
    agg[k] = 0.0
    return nothing
end

function jacc_kernel_mpnn_2!(__jacc_i, b_msg, dst, edge_feat, messages, msg_input, msg_scratch, n_edge_feat, n_edges, n_in_msg, n_msg_feat, n_node_feat, node_feat, src, w_msg)
    e = 1 + (__jacc_i - 1)
    s_node = src[e]
    d_node = dst[e]
    src_off = (s_node - 1) * n_node_feat
    dst_off = (d_node - 1) * n_node_feat
    edge_off = (e - 1) * n_edge_feat
    in_off = (e - 1) * n_in_msg
    msg_off = (e - 1) * n_msg_feat
    for k = 1:n_node_feat
        msg_input[in_off + k] = node_feat[src_off + k]
    end
    for k = 1:n_node_feat
        msg_input[in_off + n_node_feat + k] = node_feat[dst_off + k]
    end
    for k = 1:n_edge_feat
        msg_input[in_off + 2n_node_feat + k] = edge_feat[edge_off + k]
    end
    for o = 1:n_msg_feat
        s = b_msg[o]
        for i_i = 1:n_in_msg
            widx = (o - 1) * n_in_msg + i_i
            s = s + w_msg[widx] * msg_input[in_off + i_i]
        end
        msg_scratch[msg_off + o] = s
    end
    for k = 1:n_msg_feat
        msg_scratch[msg_off + k] = max(msg_scratch[msg_off + k], 0.0)
    end
    for k = 1:n_msg_feat
        messages[msg_off + k] = msg_scratch[msg_off + k]
    end
    return nothing
end

function jacc_kernel_mpnn_3!(__jacc_i, agg, dst, messages, n_edges, n_msg_feat)
    i_e = 1 + (__jacc_i - 1)
    d_node = dst[i_e]
    msg_off = (i_e - 1) * n_msg_feat
    agg_off = (d_node - 1) * n_msg_feat
    for j = 1:n_msg_feat
        Atomix.@atomic agg[agg_off + j] += messages[msg_off + j]
    end
    return nothing
end

function jacc_kernel_mpnn_4!(__jacc_i, agg, b_upd, n_in_upd, n_msg_feat, n_node_feat, n_nodes, node_feat, node_feat_out, upd_input, upd_scratch, w_upd)
    v = 1 + (__jacc_i - 1)
    node_off = (v - 1) * n_node_feat
    agg_off = (v - 1) * n_msg_feat
    uin_off = (v - 1) * n_in_upd
    for k = 1:n_node_feat
        upd_input[uin_off + k] = node_feat[node_off + k]
    end
    for k = 1:n_msg_feat
        upd_input[uin_off + n_node_feat + k] = agg[agg_off + k]
    end
    for o = 1:n_node_feat
        s = b_upd[o]
        for i_i = 1:n_in_upd
            widx = (o - 1) * n_in_upd + i_i
            s = s + w_upd[widx] * upd_input[uin_off + i_i]
        end
        upd_scratch[node_off + o] = s
    end
    for k = 1:n_node_feat
        upd_scratch[node_off + k] = max(upd_scratch[node_off + k], 0.0)
    end
    for k = 1:n_node_feat
        node_feat_out[node_off + k] = upd_scratch[node_off + k]
    end
    return nothing
end

function mpnn_jacc(node_feat, edge_feat, src, dst, w_msg, b_msg, w_upd, b_upd, n_nodes, n_edges, n_node_feat, n_edge_feat, n_msg_feat, msg_input, msg_scratch, messages, agg, upd_input, upd_scratch, node_feat_out)
    n_in_msg = 2n_node_feat + n_edge_feat
    n_in_upd = n_node_feat + n_msg_feat
    n_agg = n_nodes * n_msg_feat
    if div(n_agg - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_agg - 1, 1) + 1 jacc_kernel_mpnn_1!(agg, n_agg)
    end
    if div(n_edges - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_edges - 1, 1) + 1 jacc_kernel_mpnn_2!(b_msg, dst, edge_feat, messages, msg_input, msg_scratch, n_edge_feat, n_edges, n_in_msg, n_msg_feat, n_node_feat, node_feat, src, w_msg)
    end
    if div(n_edges - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_edges - 1, 1) + 1 jacc_kernel_mpnn_3!(agg, dst, messages, n_edges, n_msg_feat)
    end
    if div(n_nodes - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_nodes - 1, 1) + 1 jacc_kernel_mpnn_4!(agg, b_upd, n_in_upd, n_msg_feat, n_node_feat, n_nodes, node_feat, node_feat_out, upd_input, upd_scratch, w_upd)
    end
    return nothing
end
