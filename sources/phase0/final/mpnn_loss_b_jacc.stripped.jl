using CUDA
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_mpnn_loss_b_1!(__jacc_i, agg, n_agg_mpnn_c1)
    k_mpnn_c1 = 1 + (__jacc_i - 1)
    agg[k_mpnn_c1] = 0.0
    return nothing
end

function jacc_kernel_mpnn_loss_b_2!(__jacc_i, b_msg, dst, edge_feat, messages, msg_input, msg_input_stack, msg_scratch, msg_scratch_stack, n_edge_feat, n_edges, n_in_msg_mpnn_c1, n_msg_feat, n_node_feat, node_feat, src, w_msg)
    e_mpnn_c1 = 1 + (__jacc_i - 1)
    s_node_mpnn_c1 = src[e_mpnn_c1]
    d_node_mpnn_c1 = dst[e_mpnn_c1]
    src_off_mpnn_c1 = (s_node_mpnn_c1 - 1) * n_node_feat
    dst_off_mpnn_c1 = (d_node_mpnn_c1 - 1) * n_node_feat
    __icse_0 = e_mpnn_c1 - 1
    edge_off_mpnn_c1 = __icse_0 * n_edge_feat
    in_off_mpnn_c1 = __icse_0 * n_in_msg_mpnn_c1
    msg_off_mpnn_c1 = __icse_0 * n_msg_feat
    for k_mpnn_c1 = 1:n_node_feat
        __idx_msg_input_stack_0 = ((e_mpnn_c1 - 1) * (div(n_node_feat - 1, 1) + 1) + (k_mpnn_c1 - 1)) + 1
        msg_input_stack[__idx_msg_input_stack_0] = msg_input[in_off_mpnn_c1 + k_mpnn_c1]
        msg_input[in_off_mpnn_c1 + k_mpnn_c1] = node_feat[src_off_mpnn_c1 + k_mpnn_c1]
    end
    for k_mpnn_c1 = 1:n_node_feat
        __icse_1 = div(n_node_feat - 1, 1) + 1
        __idx_msg_input_stack_0 = max(0, div(n_edges - 1, 1) + 1) * max(0, __icse_1) + (((e_mpnn_c1 - 1) * __icse_1 + (k_mpnn_c1 - 1)) + 1)
        msg_input_stack[__idx_msg_input_stack_0] = msg_input[in_off_mpnn_c1 + n_node_feat + k_mpnn_c1]
        msg_input[in_off_mpnn_c1 + n_node_feat + k_mpnn_c1] = node_feat[dst_off_mpnn_c1 + k_mpnn_c1]
    end
    for k_mpnn_c1 = 1:n_edge_feat
        __icse_2 = max(0, div(n_edges - 1, 1) + 1) * max(0, div(n_node_feat - 1, 1) + 1)
        __idx_msg_input_stack_0 = (__icse_2 + __icse_2) + (((e_mpnn_c1 - 1) * (div(n_edge_feat - 1, 1) + 1) + (k_mpnn_c1 - 1)) + 1)
        msg_input_stack[__idx_msg_input_stack_0] = msg_input[in_off_mpnn_c1 + 2n_node_feat + k_mpnn_c1]
        msg_input[in_off_mpnn_c1 + 2n_node_feat + k_mpnn_c1] = edge_feat[edge_off_mpnn_c1 + k_mpnn_c1]
    end
    for o_mpnn_c1 = 1:n_msg_feat
        s_mpnn_c1 = b_msg[o_mpnn_c1]
        for i_i_mpnn_c1 = 1:n_in_msg_mpnn_c1
            widx_mpnn_c1 = (o_mpnn_c1 - 1) * n_in_msg_mpnn_c1 + i_i_mpnn_c1
            s_mpnn_c1 = s_mpnn_c1 + w_msg[widx_mpnn_c1] * msg_input[in_off_mpnn_c1 + i_i_mpnn_c1]
        end
        __idx_msg_scratch_stack_2 = ((e_mpnn_c1 - 1) * (div(n_msg_feat - 1, 1) + 1) + (o_mpnn_c1 - 1)) + 1
        msg_scratch_stack[__idx_msg_scratch_stack_2] = msg_scratch[msg_off_mpnn_c1 + o_mpnn_c1]
        msg_scratch[msg_off_mpnn_c1 + o_mpnn_c1] = s_mpnn_c1
    end
    for k_mpnn_c1 = 1:n_msg_feat
        __icse_3 = div(n_msg_feat - 1, 1) + 1
        __idx_msg_scratch_stack_0 = max(0, div(n_edges - 1, 1) + 1) * max(0, __icse_3) + (((e_mpnn_c1 - 1) * __icse_3 + (k_mpnn_c1 - 1)) + 1)
        __cse_4 = msg_scratch[msg_off_mpnn_c1 + k_mpnn_c1]
        msg_scratch_stack[__idx_msg_scratch_stack_0] = __cse_4
        msg_scratch[msg_off_mpnn_c1 + k_mpnn_c1] = max(__cse_4, 0.0)
    end
    for k_mpnn_c1 = 1:n_msg_feat
        messages[msg_off_mpnn_c1 + k_mpnn_c1] = msg_scratch[msg_off_mpnn_c1 + k_mpnn_c1]
    end
    return nothing
end

function jacc_kernel_mpnn_loss_b_3!(__jacc_i, agg, dst, messages, n_edges, n_msg_feat)
    i_e_mpnn_c1 = 1 + (__jacc_i - 1)
    d_node_mpnn_c1 = dst[i_e_mpnn_c1]
    msg_off_mpnn_c1 = (i_e_mpnn_c1 - 1) * n_msg_feat
    agg_off_mpnn_c1 = (d_node_mpnn_c1 - 1) * n_msg_feat
    for j_mpnn_c1 = 1:n_msg_feat
        Atomix.@atomic agg[agg_off_mpnn_c1 + j_mpnn_c1] += messages[msg_off_mpnn_c1 + j_mpnn_c1]
    end
    return nothing
end

function jacc_kernel_mpnn_loss_b_4!(__jacc_i, agg, b_upd, n_in_upd_mpnn_c1, n_msg_feat, n_node_feat, n_nodes, node_feat, node_feat_out, upd_input, upd_input_stack, upd_scratch, upd_scratch_stack, w_upd)
    v_mpnn_c1 = 1 + (__jacc_i - 1)
    __icse_5 = v_mpnn_c1 - 1
    node_off_mpnn_c1 = __icse_5 * n_node_feat
    agg_off_mpnn_c1 = __icse_5 * n_msg_feat
    uin_off_mpnn_c1 = __icse_5 * n_in_upd_mpnn_c1
    for k_mpnn_c1 = 1:n_node_feat
        __idx_upd_input_stack_0 = ((v_mpnn_c1 - 1) * (div(n_node_feat - 1, 1) + 1) + (k_mpnn_c1 - 1)) + 1
        upd_input_stack[__idx_upd_input_stack_0] = upd_input[uin_off_mpnn_c1 + k_mpnn_c1]
        upd_input[uin_off_mpnn_c1 + k_mpnn_c1] = node_feat[node_off_mpnn_c1 + k_mpnn_c1]
    end
    for k_mpnn_c1 = 1:n_msg_feat
        __idx_upd_input_stack_0 = max(0, div(n_nodes - 1, 1) + 1) * max(0, div(n_node_feat - 1, 1) + 1) + (((v_mpnn_c1 - 1) * (div(n_msg_feat - 1, 1) + 1) + (k_mpnn_c1 - 1)) + 1)
        upd_input_stack[__idx_upd_input_stack_0] = upd_input[uin_off_mpnn_c1 + n_node_feat + k_mpnn_c1]
        upd_input[uin_off_mpnn_c1 + n_node_feat + k_mpnn_c1] = agg[agg_off_mpnn_c1 + k_mpnn_c1]
    end
    for o_mpnn_c1 = 1:n_node_feat
        s_mpnn_c1 = b_upd[o_mpnn_c1]
        for i_i_mpnn_c1 = 1:n_in_upd_mpnn_c1
            widx_mpnn_c1 = (o_mpnn_c1 - 1) * n_in_upd_mpnn_c1 + i_i_mpnn_c1
            s_mpnn_c1 = s_mpnn_c1 + w_upd[widx_mpnn_c1] * upd_input[uin_off_mpnn_c1 + i_i_mpnn_c1]
        end
        __idx_upd_scratch_stack_2 = ((v_mpnn_c1 - 1) * (div(n_node_feat - 1, 1) + 1) + (o_mpnn_c1 - 1)) + 1
        upd_scratch_stack[__idx_upd_scratch_stack_2] = upd_scratch[node_off_mpnn_c1 + o_mpnn_c1]
        upd_scratch[node_off_mpnn_c1 + o_mpnn_c1] = s_mpnn_c1
    end
    for k_mpnn_c1 = 1:n_node_feat
        __icse_6 = div(n_node_feat - 1, 1) + 1
        __idx_upd_scratch_stack_0 = max(0, div(n_nodes - 1, 1) + 1) * max(0, __icse_6) + (((v_mpnn_c1 - 1) * __icse_6 + (k_mpnn_c1 - 1)) + 1)
        __cse_7 = upd_scratch[node_off_mpnn_c1 + k_mpnn_c1]
        upd_scratch_stack[__idx_upd_scratch_stack_0] = __cse_7
        upd_scratch[node_off_mpnn_c1 + k_mpnn_c1] = max(__cse_7, 0.0)
    end
    for k_mpnn_c1 = 1:n_node_feat
        node_feat_out[node_off_mpnn_c1 + k_mpnn_c1] = upd_scratch[node_off_mpnn_c1 + k_mpnn_c1]
    end
    return nothing
end

function jacc_kernel_mpnn_loss_b_5!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_mpnn_loss_b_6!(__jacc_i, loss, n_node_feat, n_nodes, node_feat_out, target)
    i_o = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += (node_feat_out[i_o] - target[i_o]) ^ 2
    return nothing
end

function jacc_kernel_mpnn_loss_b_7!(__jacc_i, lossb, n_node_feat, n_nodes, node_feat_out, node_feat_outb, target, targetb)
    i_o = n_nodes * n_node_feat + (__jacc_i - 1) * -1
    __cse_8 = (2 * (node_feat_out[i_o] - target[i_o])) * lossb[1]
    node_feat_outb[i_o] = node_feat_outb[i_o] + __cse_8
    targetb[i_o] = targetb[i_o] + -__cse_8
    return nothing
end

function jacc_kernel_mpnn_loss_b_8!(__jacc_i, aggb, b_updb, n_in_upd_mpnn_c1, n_msg_feat, n_node_feat, n_nodes, node_feat_outb, node_featb, upd_input, upd_input_stack, upd_inputb, upd_scratch, upd_scratch_stack, upd_scratchb, w_upd, w_updb)
    v_mpnn_c1 = n_nodes + (__jacc_i - 1) * -1
    s_mpnn_c1b = 0.0
    __icse_9 = v_mpnn_c1 - 1
    node_off_mpnn_c1 = __icse_9 * n_node_feat
    agg_off_mpnn_c1 = __icse_9 * n_msg_feat
    uin_off_mpnn_c1 = __icse_9 * n_in_upd_mpnn_c1
    for k_mpnn_c1 = n_node_feat:-1:1
        __oldb_0 = node_feat_outb[node_off_mpnn_c1 + k_mpnn_c1]
        node_feat_outb[node_off_mpnn_c1 + k_mpnn_c1] = 0.0
        upd_scratchb[node_off_mpnn_c1 + k_mpnn_c1] = upd_scratchb[node_off_mpnn_c1 + k_mpnn_c1] + __oldb_0
    end
    for k_mpnn_c1 = n_node_feat:-1:1
        __icse_10 = div(n_node_feat - 1, 1) + 1
        __idx_upd_scratch_stack_0 = max(0, div(n_nodes - 1, 1) + 1) * max(0, __icse_10) + (((v_mpnn_c1 - 1) * __icse_10 + (k_mpnn_c1 - 1)) + 1)
        upd_scratch[node_off_mpnn_c1 + k_mpnn_c1] = upd_scratch_stack[__idx_upd_scratch_stack_0]
        __oldb_2 = upd_scratchb[node_off_mpnn_c1 + k_mpnn_c1]
        upd_scratchb[node_off_mpnn_c1 + k_mpnn_c1] = 0.0
        upd_scratchb[node_off_mpnn_c1 + k_mpnn_c1] = upd_scratchb[node_off_mpnn_c1 + k_mpnn_c1] + (0.5 * (1.0 + sign(upd_scratch[node_off_mpnn_c1 + k_mpnn_c1]))) * __oldb_2
    end
    for o_mpnn_c1 = n_node_feat:-1:1
        __idx_upd_scratch_stack_0 = ((v_mpnn_c1 - 1) * (div(n_node_feat - 1, 1) + 1) + (o_mpnn_c1 - 1)) + 1
        upd_scratch[node_off_mpnn_c1 + o_mpnn_c1] = upd_scratch_stack[__idx_upd_scratch_stack_0]
        __oldb_2 = upd_scratchb[node_off_mpnn_c1 + o_mpnn_c1]
        upd_scratchb[node_off_mpnn_c1 + o_mpnn_c1] = 0.0
        s_mpnn_c1b = s_mpnn_c1b + __oldb_2
        for i_i_mpnn_c1 = n_in_upd_mpnn_c1:-1:1
            widx_mpnn_c1 = (o_mpnn_c1 - 1) * n_in_upd_mpnn_c1 + i_i_mpnn_c1
            Atomix.@atomic w_updb[widx_mpnn_c1] += upd_input[uin_off_mpnn_c1 + i_i_mpnn_c1] * s_mpnn_c1b
            upd_inputb[uin_off_mpnn_c1 + i_i_mpnn_c1] = upd_inputb[uin_off_mpnn_c1 + i_i_mpnn_c1] + w_upd[widx_mpnn_c1] * s_mpnn_c1b
        end
        __oldb_0 = s_mpnn_c1b
        s_mpnn_c1b = 0.0
        Atomix.@atomic b_updb[o_mpnn_c1] += __oldb_0
    end
    for k_mpnn_c1 = n_msg_feat:-1:1
        __idx_upd_input_stack_0 = max(0, div(n_nodes - 1, 1) + 1) * max(0, div(n_node_feat - 1, 1) + 1) + (((v_mpnn_c1 - 1) * (div(n_msg_feat - 1, 1) + 1) + (k_mpnn_c1 - 1)) + 1)
        upd_input[uin_off_mpnn_c1 + n_node_feat + k_mpnn_c1] = upd_input_stack[__idx_upd_input_stack_0]
        __oldb_2 = upd_inputb[uin_off_mpnn_c1 + n_node_feat + k_mpnn_c1]
        upd_inputb[uin_off_mpnn_c1 + n_node_feat + k_mpnn_c1] = 0.0
        aggb[agg_off_mpnn_c1 + k_mpnn_c1] = aggb[agg_off_mpnn_c1 + k_mpnn_c1] + __oldb_2
    end
    for k_mpnn_c1 = n_node_feat:-1:1
        __idx_upd_input_stack_0 = ((v_mpnn_c1 - 1) * (div(n_node_feat - 1, 1) + 1) + (k_mpnn_c1 - 1)) + 1
        upd_input[uin_off_mpnn_c1 + k_mpnn_c1] = upd_input_stack[__idx_upd_input_stack_0]
        __oldb_2 = upd_inputb[uin_off_mpnn_c1 + k_mpnn_c1]
        upd_inputb[uin_off_mpnn_c1 + k_mpnn_c1] = 0.0
        node_featb[node_off_mpnn_c1 + k_mpnn_c1] = node_featb[node_off_mpnn_c1 + k_mpnn_c1] + __oldb_2
    end
    return nothing
end

function jacc_kernel_mpnn_loss_b_9!(__jacc_i, aggb, dst, messagesb, n_edges, n_msg_feat)
    i_e_mpnn_c1 = n_edges + (__jacc_i - 1) * -1
    d_node_mpnn_c1 = dst[i_e_mpnn_c1]
    msg_off_mpnn_c1 = (i_e_mpnn_c1 - 1) * n_msg_feat
    agg_off_mpnn_c1 = (d_node_mpnn_c1 - 1) * n_msg_feat
    for j_mpnn_c1 = n_msg_feat:-1:1
        messagesb[msg_off_mpnn_c1 + j_mpnn_c1] = messagesb[msg_off_mpnn_c1 + j_mpnn_c1] + aggb[agg_off_mpnn_c1 + j_mpnn_c1]
    end
    return nothing
end

function jacc_kernel_mpnn_loss_b_10!(__jacc_i, b_msgb, dst, edge_featb, messagesb, msg_input, msg_input_stack, msg_inputb, msg_scratch, msg_scratch_stack, msg_scratchb, n_edge_feat, n_edges, n_in_msg_mpnn_c1, n_msg_feat, n_node_feat, node_featb, src, w_msg, w_msgb)
    e_mpnn_c1 = n_edges + (__jacc_i - 1) * -1
    s_mpnn_c1b = 0.0
    s_node_mpnn_c1 = src[e_mpnn_c1]
    d_node_mpnn_c1 = dst[e_mpnn_c1]
    src_off_mpnn_c1 = (s_node_mpnn_c1 - 1) * n_node_feat
    dst_off_mpnn_c1 = (d_node_mpnn_c1 - 1) * n_node_feat
    __icse_11 = e_mpnn_c1 - 1
    edge_off_mpnn_c1 = __icse_11 * n_edge_feat
    in_off_mpnn_c1 = __icse_11 * n_in_msg_mpnn_c1
    msg_off_mpnn_c1 = __icse_11 * n_msg_feat
    for k_mpnn_c1 = n_msg_feat:-1:1
        __oldb_0 = messagesb[msg_off_mpnn_c1 + k_mpnn_c1]
        messagesb[msg_off_mpnn_c1 + k_mpnn_c1] = 0.0
        msg_scratchb[msg_off_mpnn_c1 + k_mpnn_c1] = msg_scratchb[msg_off_mpnn_c1 + k_mpnn_c1] + __oldb_0
    end
    for k_mpnn_c1 = n_msg_feat:-1:1
        __icse_12 = div(n_msg_feat - 1, 1) + 1
        __idx_msg_scratch_stack_0 = max(0, div(n_edges - 1, 1) + 1) * max(0, __icse_12) + (((e_mpnn_c1 - 1) * __icse_12 + (k_mpnn_c1 - 1)) + 1)
        msg_scratch[msg_off_mpnn_c1 + k_mpnn_c1] = msg_scratch_stack[__idx_msg_scratch_stack_0]
        __oldb_2 = msg_scratchb[msg_off_mpnn_c1 + k_mpnn_c1]
        msg_scratchb[msg_off_mpnn_c1 + k_mpnn_c1] = 0.0
        msg_scratchb[msg_off_mpnn_c1 + k_mpnn_c1] = msg_scratchb[msg_off_mpnn_c1 + k_mpnn_c1] + (0.5 * (1.0 + sign(msg_scratch[msg_off_mpnn_c1 + k_mpnn_c1]))) * __oldb_2
    end
    for o_mpnn_c1 = n_msg_feat:-1:1
        __idx_msg_scratch_stack_0 = ((e_mpnn_c1 - 1) * (div(n_msg_feat - 1, 1) + 1) + (o_mpnn_c1 - 1)) + 1
        msg_scratch[msg_off_mpnn_c1 + o_mpnn_c1] = msg_scratch_stack[__idx_msg_scratch_stack_0]
        __oldb_2 = msg_scratchb[msg_off_mpnn_c1 + o_mpnn_c1]
        msg_scratchb[msg_off_mpnn_c1 + o_mpnn_c1] = 0.0
        s_mpnn_c1b = s_mpnn_c1b + __oldb_2
        for i_i_mpnn_c1 = n_in_msg_mpnn_c1:-1:1
            widx_mpnn_c1 = (o_mpnn_c1 - 1) * n_in_msg_mpnn_c1 + i_i_mpnn_c1
            Atomix.@atomic w_msgb[widx_mpnn_c1] += msg_input[in_off_mpnn_c1 + i_i_mpnn_c1] * s_mpnn_c1b
            msg_inputb[in_off_mpnn_c1 + i_i_mpnn_c1] = msg_inputb[in_off_mpnn_c1 + i_i_mpnn_c1] + w_msg[widx_mpnn_c1] * s_mpnn_c1b
        end
        __oldb_0 = s_mpnn_c1b
        s_mpnn_c1b = 0.0
        Atomix.@atomic b_msgb[o_mpnn_c1] += __oldb_0
    end
    for k_mpnn_c1 = n_edge_feat:-1:1
        __icse_13 = max(0, div(n_edges - 1, 1) + 1) * max(0, div(n_node_feat - 1, 1) + 1)
        __idx_msg_input_stack_0 = (__icse_13 + __icse_13) + (((e_mpnn_c1 - 1) * (div(n_edge_feat - 1, 1) + 1) + (k_mpnn_c1 - 1)) + 1)
        msg_input[in_off_mpnn_c1 + 2n_node_feat + k_mpnn_c1] = msg_input_stack[__idx_msg_input_stack_0]
        __oldb_2 = msg_inputb[in_off_mpnn_c1 + 2n_node_feat + k_mpnn_c1]
        msg_inputb[in_off_mpnn_c1 + 2n_node_feat + k_mpnn_c1] = 0.0
        edge_featb[edge_off_mpnn_c1 + k_mpnn_c1] = edge_featb[edge_off_mpnn_c1 + k_mpnn_c1] + __oldb_2
    end
    for k_mpnn_c1 = n_node_feat:-1:1
        __icse_14 = div(n_node_feat - 1, 1) + 1
        __idx_msg_input_stack_0 = max(0, div(n_edges - 1, 1) + 1) * max(0, __icse_14) + (((e_mpnn_c1 - 1) * __icse_14 + (k_mpnn_c1 - 1)) + 1)
        msg_input[in_off_mpnn_c1 + n_node_feat + k_mpnn_c1] = msg_input_stack[__idx_msg_input_stack_0]
        __oldb_2 = msg_inputb[in_off_mpnn_c1 + n_node_feat + k_mpnn_c1]
        msg_inputb[in_off_mpnn_c1 + n_node_feat + k_mpnn_c1] = 0.0
        Atomix.@atomic node_featb[dst_off_mpnn_c1 + k_mpnn_c1] += __oldb_2
    end
    for k_mpnn_c1 = n_node_feat:-1:1
        __idx_msg_input_stack_0 = ((e_mpnn_c1 - 1) * (div(n_node_feat - 1, 1) + 1) + (k_mpnn_c1 - 1)) + 1
        msg_input[in_off_mpnn_c1 + k_mpnn_c1] = msg_input_stack[__idx_msg_input_stack_0]
        __oldb_2 = msg_inputb[in_off_mpnn_c1 + k_mpnn_c1]
        msg_inputb[in_off_mpnn_c1 + k_mpnn_c1] = 0.0
        Atomix.@atomic node_featb[src_off_mpnn_c1 + k_mpnn_c1] += __oldb_2
    end
    return nothing
end

function jacc_kernel_mpnn_loss_b_11!(__jacc_i, aggb, n_agg_mpnn_c1)
    k_mpnn_c1 = n_agg_mpnn_c1 + (__jacc_i - 1) * -1
    aggb[k_mpnn_c1] = 0.0
    return nothing
end

function jacc_kernel_mpnn_loss_1!(__jacc_i, agg, n_agg_mpnn_c1)
    k_mpnn_c1 = 1 + (__jacc_i - 1)
    agg[k_mpnn_c1] = 0.0
    return nothing
end

function jacc_kernel_mpnn_loss_2!(__jacc_i, b_msg, dst, edge_feat, messages, msg_input, msg_scratch, n_edge_feat, n_edges, n_in_msg_mpnn_c1, n_msg_feat, n_node_feat, node_feat, src, w_msg)
    e_mpnn_c1 = 1 + (__jacc_i - 1)
    s_node_mpnn_c1 = src[e_mpnn_c1]
    d_node_mpnn_c1 = dst[e_mpnn_c1]
    src_off_mpnn_c1 = (s_node_mpnn_c1 - 1) * n_node_feat
    dst_off_mpnn_c1 = (d_node_mpnn_c1 - 1) * n_node_feat
    edge_off_mpnn_c1 = (e_mpnn_c1 - 1) * n_edge_feat
    in_off_mpnn_c1 = (e_mpnn_c1 - 1) * n_in_msg_mpnn_c1
    msg_off_mpnn_c1 = (e_mpnn_c1 - 1) * n_msg_feat
    for k_mpnn_c1 = 1:n_node_feat
        msg_input[in_off_mpnn_c1 + k_mpnn_c1] = node_feat[src_off_mpnn_c1 + k_mpnn_c1]
    end
    for k_mpnn_c1 = 1:n_node_feat
        msg_input[in_off_mpnn_c1 + n_node_feat + k_mpnn_c1] = node_feat[dst_off_mpnn_c1 + k_mpnn_c1]
    end
    for k_mpnn_c1 = 1:n_edge_feat
        msg_input[in_off_mpnn_c1 + 2n_node_feat + k_mpnn_c1] = edge_feat[edge_off_mpnn_c1 + k_mpnn_c1]
    end
    for o_mpnn_c1 = 1:n_msg_feat
        s_mpnn_c1 = b_msg[o_mpnn_c1]
        for i_i_mpnn_c1 = 1:n_in_msg_mpnn_c1
            widx_mpnn_c1 = (o_mpnn_c1 - 1) * n_in_msg_mpnn_c1 + i_i_mpnn_c1
            s_mpnn_c1 = s_mpnn_c1 + w_msg[widx_mpnn_c1] * msg_input[in_off_mpnn_c1 + i_i_mpnn_c1]
        end
        msg_scratch[msg_off_mpnn_c1 + o_mpnn_c1] = s_mpnn_c1
    end
    for k_mpnn_c1 = 1:n_msg_feat
        msg_scratch[msg_off_mpnn_c1 + k_mpnn_c1] = max(msg_scratch[msg_off_mpnn_c1 + k_mpnn_c1], 0.0)
    end
    for k_mpnn_c1 = 1:n_msg_feat
        messages[msg_off_mpnn_c1 + k_mpnn_c1] = msg_scratch[msg_off_mpnn_c1 + k_mpnn_c1]
    end
    return nothing
end

function jacc_kernel_mpnn_loss_3!(__jacc_i, agg, dst, messages, n_edges, n_msg_feat)
    i_e_mpnn_c1 = 1 + (__jacc_i - 1)
    d_node_mpnn_c1 = dst[i_e_mpnn_c1]
    msg_off_mpnn_c1 = (i_e_mpnn_c1 - 1) * n_msg_feat
    agg_off_mpnn_c1 = (d_node_mpnn_c1 - 1) * n_msg_feat
    for j_mpnn_c1 = 1:n_msg_feat
        Atomix.@atomic agg[agg_off_mpnn_c1 + j_mpnn_c1] += messages[msg_off_mpnn_c1 + j_mpnn_c1]
    end
    return nothing
end

function jacc_kernel_mpnn_loss_4!(__jacc_i, agg, b_upd, n_in_upd_mpnn_c1, n_msg_feat, n_node_feat, n_nodes, node_feat, node_feat_out, upd_input, upd_scratch, w_upd)
    v_mpnn_c1 = 1 + (__jacc_i - 1)
    node_off_mpnn_c1 = (v_mpnn_c1 - 1) * n_node_feat
    agg_off_mpnn_c1 = (v_mpnn_c1 - 1) * n_msg_feat
    uin_off_mpnn_c1 = (v_mpnn_c1 - 1) * n_in_upd_mpnn_c1
    for k_mpnn_c1 = 1:n_node_feat
        upd_input[uin_off_mpnn_c1 + k_mpnn_c1] = node_feat[node_off_mpnn_c1 + k_mpnn_c1]
    end
    for k_mpnn_c1 = 1:n_msg_feat
        upd_input[uin_off_mpnn_c1 + n_node_feat + k_mpnn_c1] = agg[agg_off_mpnn_c1 + k_mpnn_c1]
    end
    for o_mpnn_c1 = 1:n_node_feat
        s_mpnn_c1 = b_upd[o_mpnn_c1]
        for i_i_mpnn_c1 = 1:n_in_upd_mpnn_c1
            widx_mpnn_c1 = (o_mpnn_c1 - 1) * n_in_upd_mpnn_c1 + i_i_mpnn_c1
            s_mpnn_c1 = s_mpnn_c1 + w_upd[widx_mpnn_c1] * upd_input[uin_off_mpnn_c1 + i_i_mpnn_c1]
        end
        upd_scratch[node_off_mpnn_c1 + o_mpnn_c1] = s_mpnn_c1
    end
    for k_mpnn_c1 = 1:n_node_feat
        upd_scratch[node_off_mpnn_c1 + k_mpnn_c1] = max(upd_scratch[node_off_mpnn_c1 + k_mpnn_c1], 0.0)
    end
    for k_mpnn_c1 = 1:n_node_feat
        node_feat_out[node_off_mpnn_c1 + k_mpnn_c1] = upd_scratch[node_off_mpnn_c1 + k_mpnn_c1]
    end
    return nothing
end

function jacc_kernel_mpnn_loss_5!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_mpnn_loss_6!(__jacc_i, loss, n_node_feat, n_nodes, node_feat_out, target)
    i_o = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += (node_feat_out[i_o] - target[i_o]) ^ 2
    return nothing
end

function initstacks_mpnn_loss_b_jacc(n_edge_feat, n_edges, n_msg_feat, n_node_feat, n_nodes)
    msg_input_stack = JACC.zeros(Float64, (max(0, div(n_edges - 1, 1) + 1) * max(0, div(n_node_feat - 1, 1) + 1) + max(0, div(n_edges - 1, 1) + 1) * max(0, div(n_node_feat - 1, 1) + 1)) + max(0, div(n_edges - 1, 1) + 1) * max(0, div(n_edge_feat - 1, 1) + 1))
    msg_scratch_stack = JACC.zeros(Float64, max(0, div(n_edges - 1, 1) + 1) * max(0, div(n_msg_feat - 1, 1) + 1) + max(0, div(n_edges - 1, 1) + 1) * max(0, div(n_msg_feat - 1, 1) + 1))
    upd_input_stack = JACC.zeros(Float64, max(0, div(n_nodes - 1, 1) + 1) * max(0, div(n_node_feat - 1, 1) + 1) + max(0, div(n_nodes - 1, 1) + 1) * max(0, div(n_msg_feat - 1, 1) + 1))
    upd_scratch_stack = JACC.zeros(Float64, max(0, div(n_nodes - 1, 1) + 1) * max(0, div(n_node_feat - 1, 1) + 1) + max(0, div(n_nodes - 1, 1) + 1) * max(0, div(n_node_feat - 1, 1) + 1))
    return (msg_input_stack, msg_scratch_stack, upd_input_stack, upd_scratch_stack)
end

function mpnn_loss_b_jacc(node_feat, node_featb, edge_feat, edge_featb, src, dst, w_msg, w_msgb, b_msg, b_msgb, w_upd, w_updb, b_upd, b_updb, n_nodes, n_edges, n_node_feat, n_edge_feat, n_msg_feat, msg_input, msg_inputb, msg_scratch, msg_scratchb, messages, messagesb, agg, aggb, upd_input, upd_inputb, upd_scratch, upd_scratchb, node_feat_out, node_feat_outb, target, targetb, loss, lossb, msg_input_stack, msg_scratch_stack, upd_input_stack, upd_scratch_stack)
    s_mpnn_c1 = 0.0
    s_mpnn_c1b = 0.0
    n_in_msg_mpnn_c1 = 2n_node_feat + n_edge_feat
    n_in_upd_mpnn_c1 = n_node_feat + n_msg_feat
    n_agg_mpnn_c1 = n_nodes * n_msg_feat
    if div(n_agg_mpnn_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_agg_mpnn_c1 - 1, 1) + 1 jacc_kernel_mpnn_loss_b_1!(agg, n_agg_mpnn_c1)
    end
    if div(n_edges - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_edges - 1, 1) + 1 jacc_kernel_mpnn_loss_b_2!(b_msg, dst, edge_feat, messages, msg_input, msg_input_stack, msg_scratch, msg_scratch_stack, n_edge_feat, n_edges, n_in_msg_mpnn_c1, n_msg_feat, n_node_feat, node_feat, src, w_msg)
    end
    if div(n_edges - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_edges - 1, 1) + 1 jacc_kernel_mpnn_loss_b_3!(agg, dst, messages, n_edges, n_msg_feat)
    end
    if div(n_nodes - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_nodes - 1, 1) + 1 jacc_kernel_mpnn_loss_b_4!(agg, b_upd, n_in_upd_mpnn_c1, n_msg_feat, n_node_feat, n_nodes, node_feat, node_feat_out, upd_input, upd_input_stack, upd_scratch, upd_scratch_stack, w_upd)
    end
    if div(n_nodes * n_node_feat - 1, 1) + 1 < 32768
        if div(n_nodes * n_node_feat - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_nodes * n_node_feat - 1, 1) + 1 jacc_kernel_mpnn_loss_b_6!(loss, n_node_feat, n_nodes, node_feat_out, target)
        end
    else
        if div(n_nodes * n_node_feat - 1, 1) + 1 > 0
            __jgen_redval_5 = JACC.@parallel_reduce(range = div(n_nodes * n_node_feat - 1, 1) + 1, (((i_o, node_feat_out, target)->(node_feat_out[i_o] - target[i_o]) ^ 2))(node_feat_out, target))
            JACC.@parallel_for range = 1 jacc_kernel_mpnn_loss_b_5!(loss, __jgen_redval_5)
        end
    end
    n_in_msg_mpnn_c1 = 2n_node_feat + n_edge_feat
    n_in_upd_mpnn_c1 = n_node_feat + n_msg_feat
    n_agg_mpnn_c1 = n_nodes * n_msg_feat
    if div(1 - n_nodes * n_node_feat, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_nodes * n_node_feat, -1) + 1 jacc_kernel_mpnn_loss_b_7!(lossb, n_node_feat, n_nodes, node_feat_out, node_feat_outb, target, targetb)
    end
    if div(1 - n_nodes, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_nodes, -1) + 1 jacc_kernel_mpnn_loss_b_8!(aggb, b_updb, n_in_upd_mpnn_c1, n_msg_feat, n_node_feat, n_nodes, node_feat_outb, node_featb, upd_input, upd_input_stack, upd_inputb, upd_scratch, upd_scratch_stack, upd_scratchb, w_upd, w_updb)
    end
    if div(1 - n_edges, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_edges, -1) + 1 jacc_kernel_mpnn_loss_b_9!(aggb, dst, messagesb, n_edges, n_msg_feat)
    end
    if div(1 - n_edges, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_edges, -1) + 1 jacc_kernel_mpnn_loss_b_10!(b_msgb, dst, edge_featb, messagesb, msg_input, msg_input_stack, msg_inputb, msg_scratch, msg_scratch_stack, msg_scratchb, n_edge_feat, n_edges, n_in_msg_mpnn_c1, n_msg_feat, n_node_feat, node_featb, src, w_msg, w_msgb)
    end
    if div(1 - n_agg_mpnn_c1, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_agg_mpnn_c1, -1) + 1 jacc_kernel_mpnn_loss_b_11!(aggb, n_agg_mpnn_c1)
    end
    return nothing
end

function mpnn_loss_jacc(node_feat, edge_feat, src, dst, w_msg, b_msg, w_upd, b_upd, n_nodes, n_edges, n_node_feat, n_edge_feat, n_msg_feat, msg_input, msg_scratch, messages, agg, upd_input, upd_scratch, node_feat_out, target, loss)
    n_in_msg_mpnn_c1 = 2n_node_feat + n_edge_feat
    n_in_upd_mpnn_c1 = n_node_feat + n_msg_feat
    n_agg_mpnn_c1 = n_nodes * n_msg_feat
    if div(n_agg_mpnn_c1 - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_agg_mpnn_c1 - 1, 1) + 1 jacc_kernel_mpnn_loss_1!(agg, n_agg_mpnn_c1)
    end
    if div(n_edges - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_edges - 1, 1) + 1 jacc_kernel_mpnn_loss_2!(b_msg, dst, edge_feat, messages, msg_input, msg_scratch, n_edge_feat, n_edges, n_in_msg_mpnn_c1, n_msg_feat, n_node_feat, node_feat, src, w_msg)
    end
    if div(n_edges - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_edges - 1, 1) + 1 jacc_kernel_mpnn_loss_3!(agg, dst, messages, n_edges, n_msg_feat)
    end
    if div(n_nodes - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_nodes - 1, 1) + 1 jacc_kernel_mpnn_loss_4!(agg, b_upd, n_in_upd_mpnn_c1, n_msg_feat, n_node_feat, n_nodes, node_feat, node_feat_out, upd_input, upd_scratch, w_upd)
    end
    if div(n_nodes * n_node_feat - 1, 1) + 1 < 32768
        if div(n_nodes * n_node_feat - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(n_nodes * n_node_feat - 1, 1) + 1 jacc_kernel_mpnn_loss_6!(loss, n_node_feat, n_nodes, node_feat_out, target)
        end
    else
        if div(n_nodes * n_node_feat - 1, 1) + 1 > 0
            __jgen_redval_5 = JACC.@parallel_reduce(range = div(n_nodes * n_node_feat - 1, 1) + 1, (((i_o, node_feat_out, target)->(node_feat_out[i_o] - target[i_o]) ^ 2))(node_feat_out, target))
            JACC.@parallel_for range = 1 jacc_kernel_mpnn_loss_5!(loss, __jgen_redval_5)
        end
    end
    return nothing
end
