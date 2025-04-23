
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from modules import RMSNorm, ParallelGatedMLP


# knn pomo
class GTSPGIMFModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        
        # 图模态编码器
        self.graph_encoder = TSP_Encoder(**model_params)
        
        # 图像模态编码器
        from ImageEncoder import VisionTransformer, CoordinateImageBuilder
        self.image_builder = CoordinateImageBuilder(**model_params)
        self.vision_encoder = VisionTransformer(**model_params)
        
        # 多模态融合层
        self.multimodal_fusion = MultimodalFusionLayer(**model_params)
        
        # 解码器
        self.decoder = Decoder(**model_params)
        
        # 编码节点和图嵌入缓存
        self.encoded_nodes = None
        self.graph_embedding = None
        self.image_embedding = None
        self.fused_nodes = None

    def pre_forward(self, reset_state):
        node_xy = reset_state.node_xy
        cluster_idx = reset_state.cluster_idx
        
        # 图编码
        self.encoded_nodes = self.graph_encoder(node_xy, cluster_idx)
        self.graph_embedding = self.encoded_nodes.mean(-2)
        
        # 图像编码
        batch_size = node_xy.shape[0]
        coord_image = self.image_builder.build_gtsp_image(node_xy, cluster_idx)
        self.image_embedding = self.vision_encoder(coord_image)
        
        # 多模态融合
        self.fused_nodes = self.multimodal_fusion(
            self.encoded_nodes, 
            self.image_embedding,
            self.graph_embedding
        )
        
        # 设置解码器键值
        self.decoder.set_kv(self.fused_nodes)

    def get_k_nearest_neighbor(self, node_xy, pomo_size):
        k_nn = (node_xy[:, 0][:, None, :].expand_as(node_xy) - node_xy).norm(p=2, dim=-1).sort(dim=-1)[1]
        return k_nn[:, 1:pomo_size+1]

    def forward(self, state, node_xy):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long, device=node_xy.device)
            prob_node = torch.ones(size=(batch_size, pomo_size), device=node_xy.device)
            # action = torch.zeros(size=(batch_size, pomo_size, 2))

        elif state.selected_count == 1:  # Second Move, POMO
            k_neigbors = self.get_k_nearest_neighbor(node_xy, pomo_size)
            selected = k_neigbors
            prob_node = torch.ones(size=(batch_size, pomo_size), device=node_xy.device)
        else:
            encoded_last_node = _get_encoding(self.fused_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            fused_graph_embedding = self.graph_embedding + 0.3 * torch.mean(self.image_embedding, dim=1)
            probs_node = self.decoder(encoded_last_node, 
                                      fused_graph_embedding[:, None, :].expand(batch_size, pomo_size, -1),
                                      ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)
            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:
                    selected = probs_node.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    prob_node = probs_node[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    if (prob_node != 0).all():
                        break


            else:
                selected = probs_node.argmax(dim=2)
                # shape: (batch, pomo)
                prob_node = None

        return selected, prob_node


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_node = nn.Linear(2, embedding_dim)
        self.embedding_cluster_idx = nn.Linear(1, embedding_dim)
        self.embedding_all = nn.Linear(2*embedding_dim, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, node_xy, cluster_idx):
        # data.shape: (batch, problem, 2)

        embedded_node = self.embedding_node(node_xy)
        embedded_cluster_idx = self.embedding_cluster_idx(cluster_idx.float()[:, :, None])
        # shape: (batch, problem, embedding)

        out = self.embedding_all(torch.cat((embedded_node, embedded_cluster_idx), -1))
        for layer in self.layers:
            out = layer(out)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.norm_attn = RMSNorm(embedding_dim, affine=True, track_running_stats=False)
        self.norm_ffn = RMSNorm(embedding_dim, affine=True, track_running_stats=False)
        self.feedForward = ParallelGatedMLP()

    def forward(self, input):

        input1 = self.norm_attn(input)

        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        # out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        # multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)
        out = F.scaled_dot_product_attention(q, k, v)
        multi_head_out = self.multi_head_combine(rearrange(out, "b h s d -> b s (h d)"))

        out1 = input + multi_head_out
        out_norm = self.norm_ffn(out1)
        out2 = self.feedForward(out_norm)
        out3 = out1 + out2

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


########################################
# DECODER
########################################

class Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_graph = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    # def set_gc(self, encoded_nodes):
    #     # encoded_nodes.shape: (batch, problem, embedding)  # n can be 1 or pomo
    #     head_num = self.model_params['head_num']
    #     self.gc = reshape_by_heads(self.Wq_1(encoded_nodes.mean(-2)[:, None, :]), head_num=head_num)
    #     # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, graph_embedding, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        q_graph = reshape_by_heads(self.Wq_graph(graph_embedding), head_num=head_num)
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        q = q_last + q_graph
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


# def new_reshape_by_heads(qkv, head_num):
#     # q.shape: (batch, pomo_size, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE
#
#     batch_s, ps, n, _ = qkv.size()
#
#     q_reshaped = qkv.reshape(batch_s, ps, n, head_num, -1)
#     # shape: (batch, ps, n, head_num, key_dim)
#
#     q_transposed = q_reshaped.view(batch_s, head_num, ps, n, -1)
#     # shape: (batch, head_num, n, key_dim)
#
#     return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = RMSNorm(embedding_dim, affine=True, track_running_stats=False)
        # self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input):


        normalized = self.norm(input)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


from modules import RMSNorm, ParallelGatedMLP


class MultimodalFusionLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        
        # 多模态融合层数
        self.fusion_layer_num = model_params.get('fusion_layer_num', 3)
        
        # 模态特定瓶颈数量
        self.nb = model_params.get('bottleneck_size', 10)
        
        # 图模态瓶颈
        self.graph_bottlenecks = nn.Parameter(
            torch.randn(1, self.nb, embedding_dim)
        )
        
        # 图像模态瓶颈
        self.image_bottlenecks = nn.Parameter(
            torch.randn(1, self.nb, embedding_dim)
        )
        
        # 多模态层
        self.layers = nn.ModuleList([
            MultimodalFusionSingleLayer(**model_params) 
            for _ in range(self.fusion_layer_num)
        ])
    
    def forward(self, graph_nodes, image_nodes, graph_embedding):
        batch_size = graph_nodes.size(0)
        
        # 扩展瓶颈到批次大小
        graph_bottlenecks = self.graph_bottlenecks.expand(batch_size, -1, -1)
        image_bottlenecks = self.image_bottlenecks.expand(batch_size, -1, -1)
        
        for layer in self.layers:
            graph_nodes, graph_bottlenecks, image_nodes, image_bottlenecks = layer(
                graph_nodes, graph_bottlenecks, 
                image_nodes, image_bottlenecks
            )
        
        # 融合两种模态的节点表示
        fused_nodes = graph_nodes + 0.5 * image_nodes.mean(dim=1, keepdim=True).expand_as(graph_nodes)
        
        return fused_nodes
    

class MultimodalFusionSingleLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']
        
        # 图引导的跨注意力
        self.graph_guided_cross_attn = MultiHeadCrossAttention(
            embedding_dim, head_num, qkv_dim
        )
        
        # 图像引导的跨注意力
        self.image_guided_cross_attn = MultiHeadCrossAttention(
            embedding_dim, head_num, qkv_dim
        )
        
        # 规范化层
        self.norm_graph = nn.LayerNorm(embedding_dim)
        self.norm_image = nn.LayerNorm(embedding_dim)
        
        # 前馈网络
        self.ff_graph = ParallelGatedMLP(hidden_size=embedding_dim)
        self.norm_ff_graph = nn.LayerNorm(embedding_dim)
        
        self.ff_image = ParallelGatedMLP(hidden_size=embedding_dim)
        self.norm_ff_image = nn.LayerNorm(embedding_dim)
    
    def forward(self, graph_nodes, graph_bottlenecks, image_nodes, image_bottlenecks):
        # 图引导的跨注意力
        graph_input = torch.cat([graph_nodes, graph_bottlenecks], dim=1)
        image_keys = torch.cat([image_nodes, image_bottlenecks], dim=1)
        
        graph_attn_out = self.graph_guided_cross_attn(
            graph_input, image_keys, image_keys
        )
        
        # 分离节点和瓶颈
        updated_graph_nodes = graph_attn_out[:, :graph_nodes.size(1)]
        updated_graph_bottlenecks = graph_attn_out[:, graph_nodes.size(1):]
        
        # 规范化和残差连接
        normalized_graph_nodes = self.norm_graph(graph_nodes + updated_graph_nodes)
        normalized_graph_bottlenecks = self.norm_graph(graph_bottlenecks + updated_graph_bottlenecks)
        
        # 前馈网络
        ff_graph_nodes = normalized_graph_nodes + self.ff_graph(normalized_graph_nodes)
        ff_graph_bottlenecks = normalized_graph_bottlenecks + self.ff_graph(normalized_graph_bottlenecks)
        
        # 同样的操作应用于图像模态
        image_input = torch.cat([image_nodes, image_bottlenecks], dim=1)
        graph_keys = torch.cat([graph_nodes, graph_bottlenecks], dim=1)
        
        image_attn_out = self.image_guided_cross_attn(
            image_input, graph_keys, graph_keys
        )
        
        updated_image_nodes = image_attn_out[:, :image_nodes.size(1)]
        updated_image_bottlenecks = image_attn_out[:, image_nodes.size(1):]
        
        normalized_image_nodes = self.norm_image(image_nodes + updated_image_nodes)
        normalized_image_bottlenecks = self.norm_image(image_bottlenecks + updated_image_bottlenecks)
        
        ff_image_nodes = normalized_image_nodes + self.ff_image(normalized_image_nodes)
        ff_image_bottlenecks = normalized_image_bottlenecks + self.ff_image(normalized_image_bottlenecks)
        
        return ff_graph_nodes, ff_graph_bottlenecks, ff_image_nodes, ff_image_bottlenecks
    

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embedding_dim, head_num, qkv_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
    
    def forward(self, q, k, v):
        q = reshape_by_heads(self.Wq(q), head_num=self.head_num)
        k = reshape_by_heads(self.Wk(k), head_num=self.head_num)
        v = reshape_by_heads(self.Wv(v), head_num=self.head_num)
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.multi_head_combine(out)
        
        return out




