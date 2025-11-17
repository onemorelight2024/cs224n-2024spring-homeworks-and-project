'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''
'''
extension部分：
    1.AggregationResult结构体，用于保存损失值、梯度、参数。
    2.实现aggregate_task_losses函数，用于多任务损失聚合。这个函数是计算损失的总接口，包含了平均值、加权、gradient surgery三种方法。
    3.实现surgery分支，用于实现gradient surgery。输入：AggregationResult结构体。输出：AggregationResult结构体。输出：损失值的向量形式。
    4.主函数中修改逻辑。

    3.实现compute_siar_penalty函数，用于平滑诱导对抗正则化。
    4.训练函数中对损失加入compute_siar_penalty，包含输入args逻辑

    5.修改三个任务的具体实现，包含新加入一个投影层，并因此修改输出层为1个神经元。

    6.更新readme。
'''

import random, numpy as np, argparse
from typing import Dict, Optional, List
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_sts,model_eval_multitask, model_eval_test_multitask


TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

class AggregationResult:
    #因为我们不能用自动反向传播了，所以需要手动计算梯度。
    #所以需要保存损失值、梯度、参数。
    loss:torch.Tensor
    grads:Optional[List[torch.Tensor]] = None
    params:Optional[List[torch.nn.Parameter]] = None
def aggregate_task_losses(model, losses_dic, strategy='average', weight_map_for_weighted=None, weight_map_for_gradient_surgery=None, global_step=0):
    '''
    losses_dic: dict[str, torch.Tensor]
    '''
    if strategy=='average':
        return sum(loss.mean() for loss in losses_dic.values())/len(losses_dic)
    elif strategy=='weighted':
        return sum(weight_map_for_weighted[task_name]*loss.mean() for task_name, loss in losses_dic.items())
    #此处返回的是梯度，而不是损失值，需要在主函数里
    elif strategy=='gradient_surgery':
        return _apply_pcgrad(model,losses_dic,weight_map_for_gradient_surgery)
def _apply_pcgrad(model,losses_dic,weight_map):
    #1.收集共享可训练参数（只收集需要梯度的参数）
    params=[p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise ValueError("No parameters require gradients!")

    #2.计算梯度，保存为List[List[tensor]]形式，内层列表为一个任务
    #保证grads[index1][x]与grads[index2][x]是同一个参数的梯度,以直接做点积判断。
    grads=[]
    for task_name, loss in losses_dic.items():
        if task_name not in weight_map:
            continue
        # 确保loss有梯度
        if not loss.requires_grad:
            raise ValueError(f"Loss for task {task_name} does not require grad!")
        weighted_loss = loss * weight_map[task_name]
        # allow_unused=True: 不同任务使用不同的参数子集（例如sentiment任务不使用paraphrase_head）
        # 即使full-model模式下，某些参数在特定任务的loss计算中可能未被使用
        grad = torch.autograd.grad(weighted_loss, params, retain_graph=True, allow_unused=True)
        # 将None替换为零梯度（对于未使用的参数）
        grad = tuple(g if g is not None else torch.zeros_like(p) for g, p in zip(grad, params))
        grads.append(grad)
    #3.对每个任务进行循环，找到梯度冲突的部分
    for i in range(len(grads)):
        compare_index=list(range(len(grads)))
        grads_compare=random.sample(compare_index, len(compare_index))
        for j in range(len(grads_compare)):
            if j==i:
                continue
            else:
                dot = sum((gi_elem * gj_elem).sum() for gi_elem, gj_elem in zip(grads[i], grads[j]))
                if dot < 0:
                    # 计算grads[j]的L2范数的平方
                    grad_i_norm_sq = sum(gi_elem.pow(2).sum() for gi_elem in grads[i])
                    if grad_i_norm_sq > 0:
                        # 逐个元素更新梯度，这里的gj_elem是grads[j]的第j个元素，gi_elem是grads[i]的第i个元素。形状都为整个的矩阵
                        grads[j] = tuple(gj_elem - gi_elem * dot / grad_i_norm_sq 
                                        for gi_elem, gj_elem in zip(grads[i], grads[j]))
    #记得合并梯度（只合并需要梯度的参数）
    merged_grads = []
    for param_idx in range(len(params)):
        merged_grad = sum(g[param_idx].detach() for g in grads) / len(grads)
        merged_grads.append(merged_grad)
    return merged_grads


def compute_siar_penalty(model: nn.Module,
                         input_ids: torch.Tensor,
                         attention_mask: torch.Tensor,
                         input_ids_2: torch.Tensor,
                         attention_mask_2: torch.Tensor,
                         task: str,
                         device: torch.device,
                         epsilon: float,
                         labels: torch.Tensor) -> torch.Tensor:
    """
    平滑诱导对抗正则化（Smoothness-Inducing Adversarial Regularization）框架。
    TODO: 在此函数内实现具体的对抗扰动生成与平滑正则项计算。
    当前返回0作为占位，便于后续自行填充实现。
    """
    if task=='sentiment':
        input_ids=input_ids.to(device)
        attention_mask=attention_mask.to(device)
        #为什么要加model.bert.embed。因为是两层封装。model是我们的multitask类的实例，而model内部的编码器是一个基础的bert结构，即self.bert。
        embedding_output=model.bert.embed(input_ids)
        #生成扰动
        noise = torch.randn_like(embedding_output) * epsilon
        noise.requires_grad_(True)
        res=model.bert.forward_with_embed(embedding_output+noise, attention_mask)
        #一阶近似
        res_logits=model.sst_pre(res)
        # labels是外部传入的logits（常数），res_logits需要梯度
        loss_kl = F.mse_loss(labels.softmax(dim=-1), res_logits.softmax(dim=-1), reduction='sum')
        # 计算梯度，不需要create_graph和retain_graph（因为labels是常数）
        noise_grad=torch.autograd.grad(loss_kl, noise, create_graph=False, retain_graph=False)[0]
        # 释放中间变量
        del res, res_logits, loss_kl
        # 更新noise，detach梯度值避免保留不必要的计算图
        noise=noise+noise_grad.detach()
        del noise_grad
        # 用更新后的noise重新计算（保留embedding_output避免重复计算）
        res=model.bert.forward_with_embed(embedding_output+noise, attention_mask)
        del noise, embedding_output
        res_logitss=model.sst_pre(res)
        del res
        # 最终的loss需要保留对模型参数的梯度，注意kl散度第一个是log_softmax
        loss_kl=F.kl_div(res_logitss.log_softmax(dim=-1), labels.softmax(dim=-1), reduction='sum')
        return loss_kl.mean()

    elif task=='paraphrase':
        input_ids_1 = input_ids.to(device)
        attention_mask_1 = attention_mask.to(device)
        input_ids_2 = input_ids_2.to(device)
        attention_mask_2 = attention_mask_2.to(device)
        embedding_output_1=model.bert.embed(input_ids_1)
        embedding_output_2=model.bert.embed(input_ids_2)
        noise_1 = torch.randn_like(embedding_output_1) * epsilon
        noise_2 = torch.randn_like(embedding_output_2) * epsilon
        noise_1.requires_grad_(True)
        noise_2.requires_grad_(True)
        res_1=model.bert.forward_with_embed(embedding_output_1+noise_1, attention_mask_1)
        res_2=model.bert.forward_with_embed(embedding_output_2+noise_2, attention_mask_2)
        #一阶近似
        res_logits=model.par_pre(res_1,res_2)
        # Paraphrase任务输出是单个logit，使用MSE而不是KL散度
        loss_kl=F.mse_loss(torch.sigmoid(labels), torch.sigmoid(res_logits), reduction='sum')
        # 一次性计算两个梯度，避免计算图被提前释放
        noise1_grad, noise2_grad = torch.autograd.grad(loss_kl, [noise_1, noise_2], create_graph=False, retain_graph=False)
        # 释放中间变量
        del res_1, res_2, res_logits, loss_kl
        # 更新noise，detach梯度值避免保留不必要的计算图
        noise_1=noise_1+noise1_grad.detach()
        noise_2=noise_2+noise2_grad.detach()
        del noise1_grad, noise2_grad
        # 用更新后的noise重新计算（保留embedding_output避免重复计算）
        res_1=model.bert.forward_with_embed(embedding_output_1+noise_1, attention_mask_1)
        res_2=model.bert.forward_with_embed(embedding_output_2+noise_2, attention_mask_2)
        del noise_1, noise_2, embedding_output_1, embedding_output_2
        res_logitss=model.par_pre(res_1,res_2)
        del res_1, res_2
        loss_kl=F.mse_loss(torch.sigmoid(labels), torch.sigmoid(res_logitss), reduction='sum')
        return loss_kl.mean()
    elif task=='similarity':
        input_ids_1 = input_ids.to(device)
        attention_mask_1 = attention_mask.to(device)
        input_ids_2 = input_ids_2.to(device)
        attention_mask_2 = attention_mask_2.to(device)
        embedding_output_1=model.bert.embed(input_ids_1)
        embedding_output_2=model.bert.embed(input_ids_2)
        noise_1 = torch.randn_like(embedding_output_1) * epsilon
        noise_2 = torch.randn_like(embedding_output_2) * epsilon
        noise_1.requires_grad_(True)
        noise_2.requires_grad_(True)
        res_1=model.bert.forward_with_embed(embedding_output_1+noise_1, attention_mask_1)
        res_2=model.bert.forward_with_embed(embedding_output_2+noise_2, attention_mask_2)
        #一阶近似
        res_logits=model.sim_pre(res_1,res_2)
        # Similarity任务输出是单个值（回归），使用MSE而不是KL散度
        loss_kl=F.mse_loss(labels, res_logits, reduction='sum')
        # 一次性计算两个梯度，避免计算图被提前释放
        noise1_grad, noise2_grad = torch.autograd.grad(loss_kl, [noise_1, noise_2], create_graph=False, retain_graph=False)
        del res_1, res_2, res_logits, loss_kl
        # 更新noise，detach梯度值避免保留不必要的计算图
        noise_1=noise_1+noise1_grad.detach()
        noise_2=noise_2+noise2_grad.detach()
        del noise1_grad, noise2_grad
        # 用更新后的noise重新计算（保留embedding_output避免重复计算）
        res_1=model.bert.forward_with_embed(embedding_output_1+noise_1, attention_mask_1)
        res_2=model.bert.forward_with_embed(embedding_output_2+noise_2, attention_mask_2)
        del noise_1, noise_2, embedding_output_1, embedding_output_2
        res_logitss=model.sim_pre(res_1,res_2)
        del res_1, res_2
        loss_kl=F.mse_loss(labels, res_logitss, reduction='sum')
        return loss_kl.mean()


class MultitaskBERT(nn.Module):
    #分类和预训练时优先使用[CLS]
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained('./bert-base-uncased')
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        projection_dim = getattr(config, "task_projection_dim", BERT_HIDDEN_SIZE)
        self.sentiment_projection = nn.Linear(BERT_HIDDEN_SIZE, projection_dim)
        self.sentiment_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_head = nn.Linear(projection_dim, N_SENTIMENT_CLASSES)

        paraphrase_dim = getattr(config, "paraphrase_projection_dim", projection_dim)
        self.paraphrase_projection = nn.Linear(BERT_HIDDEN_SIZE * 16, paraphrase_dim)
        self.paraphrase_head = nn.Linear(paraphrase_dim, 1)

        similarity_dim = getattr(config, "similarity_projection_dim", projection_dim)
        self.similarity_projection = nn.Linear(BERT_HIDDEN_SIZE * 4, similarity_dim)
        self.similarity_head = nn.Linear(similarity_dim, 1)


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        
        #创新点1
        return self.bert(input_ids,attention_mask)


    def predict_sentiment(self, input_ids, attention_mask, task='sentiment'):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        w=self.forward(input_ids,attention_mask)
        pooled = w['pooler_output']
        projected = torch.tanh(self.sentiment_projection(pooled))
        output = self.sentiment_dropout(projected)
        return self.sentiment_head(output)
    def sst_pre(self, w):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO

        pooled = w['pooler_output']
        projected = torch.tanh(self.sentiment_projection(pooled))
        output = self.sentiment_dropout(projected)
        return self.sentiment_head(output)
    def par_pre(self, w1,w2):
        '''
        没有归一化
        '''
        ### TODO
        H_a=w1['last_hidden_state']
        H_b=w2['last_hidden_state']
        #这里需要句子级别的向量
        S = torch.matmul(H_a, H_b.transpose(1, 2))     # [B, T_a, T_b]
        attention_weights_for_b = torch.softmax(S, dim=2)  # a矩阵的每个词在b矩阵中的对齐权重(softmax的是b)
        attention_weights_for_a = torch.softmax(S.transpose(1, 2), dim=2)  # b矩阵的每个词在a矩阵中的对齐权重
        # 释放S，因为已经不需要了
        del S
        H_b_aligned = torch.matmul(attention_weights_for_b, H_b)  # 包含b矩阵信息的a矩阵
        H_a_aligned = torch.matmul(attention_weights_for_a, H_a)  # 包含a矩阵信息的b矩阵
        # 释放attention_weights，因为已经不需要了
        del attention_weights_for_b, attention_weights_for_a
        # 计算中间结果并立即使用，避免存储大张量
        H_a_diff = torch.abs(H_a - H_b_aligned)
        H_a_mul = H_a * H_b_aligned
        A = torch.cat([H_a, H_b_aligned, H_a_diff, H_a_mul], dim=-1)#元素相乘反应一致性
        del H_a_diff, H_a_mul
        H_b_diff = torch.abs(H_b - H_a_aligned)
        H_b_mul = H_b * H_a_aligned
        B = torch.cat([H_b, H_a_aligned, H_b_diff, H_b_mul], dim=-1)
        del H_b_diff, H_b_mul, H_a, H_b, H_b_aligned, H_a_aligned
        # 提取特征后立即释放A和B
        A_mean = A.mean(dim=1)         # [B, 4H]
        A_max, _ = A.max(dim=1)
        B_mean = B.mean(dim=1)
        B_max, _ = B.max(dim=1)
        del A, B

        features = torch.cat([A_mean, A_max, B_mean, B_max], dim=1)  # [B, 16H]
        del A_mean, A_max, B_mean, B_max
        projected = torch.tanh(self.paraphrase_projection(features))
        logits = self.paraphrase_head(projected).squeeze(-1)
        
        return logits
    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        H_a=self.forward(input_ids_1,attention_mask_1)['last_hidden_state']
        H_b=self.forward(input_ids_2,attention_mask_2)['last_hidden_state']
        #这里需要句子级别的向量
        S = torch.matmul(H_a, H_b.transpose(1, 2))     # [B, T_a, T_b]
        attention_weights_for_b = torch.softmax(S, dim=2)  # a矩阵的每个词在b矩阵中的对齐权重(softmax的是b)
        attention_weights_for_a = torch.softmax(S.transpose(1, 2), dim=2)  # b矩阵的每个词在a矩阵中的对齐权重
        H_b_aligned = torch.matmul(attention_weights_for_b, H_b)  # 包含b矩阵信息的a矩阵
        H_a_aligned = torch.matmul(attention_weights_for_a, H_a)  # 包含a矩阵信息的b矩阵
        A = torch.cat([H_a, H_b_aligned, torch.abs(H_a - H_b_aligned), H_a * H_b_aligned], dim=-1)#元素相乘反应一致性
        B = torch.cat([H_b, H_a_aligned, torch.abs(H_b - H_a_aligned), H_b * H_a_aligned], dim=-1)
        A_mean = A.mean(dim=1)         # [B, 4H]
        A_max, _ = A.max(dim=1)
        B_mean = B.mean(dim=1)
        B_max, _ = B.max(dim=1)

        features = torch.cat([A_mean, A_max, B_mean, B_max], dim=1)  # [B, 16H]
        projected = torch.tanh(self.paraphrase_projection(features))
        logits = self.paraphrase_head(projected).squeeze(-1)
        
        return logits
        
    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        outputs_a = self.forward(input_ids_1,attention_mask_1)['pooler_output']
        outputs_b = self.forward(input_ids_2,attention_mask_2)['pooler_output']
        cosine_score = torch.cosine_similarity(outputs_a, outputs_b, dim=-1, eps=1e-8)
        similarity_features = torch.cat([
            outputs_a,
            outputs_b,
            torch.abs(outputs_a - outputs_b),
            outputs_a * outputs_b
        ], dim=-1)
        projected = torch.tanh(self.similarity_projection(similarity_features))
        logits = self.similarity_head(projected).squeeze(-1)

        # 默认返回线性层输出；如需使用余弦分数，可在外部切换。
        return logits

    def sim_pre(self, w1,w2):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        outputs_a = w1['pooler_output']
        outputs_b = w2['pooler_output']
        cosine_score = torch.cosine_similarity(outputs_a, outputs_b, dim=-1, eps=1e-8)
        similarity_features = torch.cat([
            outputs_a,
            outputs_b,
            torch.abs(outputs_a - outputs_b),
            outputs_a * outputs_b
        ], dim=-1)
        projected = torch.tanh(self.similarity_projection(similarity_features))
        logits = self.similarity_head(projected).squeeze(-1)
        return logits

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)
    # 计算每个数据集的大小（batch 数量）
    sst_batches = len(sst_train_dataloader)
    para_batches = len(para_train_dataloader)
    sts_batches = len(sts_train_dataloader)
    # 暂时只使用sst和sts两个任务
    total_batches = sst_batches + sts_batches  # 暂时不包含para
    
    # 计算每个任务的采样权重（按数据集大小比例）
    task_weights = {
        'sentiment': sst_batches / total_batches,
        # 'paraphrase': para_batches / total_batches,  # 暂时注释掉para任务
        'similarity': sts_batches / total_batches
    }
    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode,
              'task_projection_dim': args.task_proj_dim,
              'paraphrase_projection_dim': args.paraphrase_proj_dim or args.task_proj_dim,
              'similarity_projection_dim': args.similarity_proj_dim or args.task_proj_dim,
              'use_siar': args.use_siar,
              'siar_eps': args.siar_eps,
              'siar_coeff': args.siar_coeff,
              'loss_strategy': args.loss_strategy,
              'loss_weights_spec': args.loss_weights}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0
    global_step = 0
    # 在第481-485行，确保权重是浮点数
    weight_map_for_gradient_surgery = {
        'sentiment': float(sst_batches) / float(total_batches),
        # 'paraphrase': float(para_batches) / float(total_batches),  # 暂时注释掉para任务
        'similarity': float(sts_batches) / float(total_batches)
    }

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        #采样,采样方法为搞一个任务名列表
        sst_iter=iter(sst_train_dataloader)
        # para_iter=iter(para_train_dataloader)  # 暂时注释掉，因为不使用para任务
        sts_iter=iter(sts_train_dataloader)
        # 由于使用last_task_name交替执行，只需要一个正确长度的数组来控制循环次数
        # 内容不重要，因为实际执行逻辑由last_task_name决定
        min_batches = min(sst_batches, sts_batches)
        task_names = [None] * ( min_batches)  # 创建长度为2*min_batches的数组
        
        # 调试信息
        print(f"Epoch {epoch}: sst_batches={sst_batches}, sts_batches={sts_batches}, min_batches={min_batches}, task_names_length={len(task_names)}")

        loss_components = {}
        last_task_name='sts'
        sst_count = 0  # 调试：统计 sst_iter 被调用的次数
        sts_count = 0  # 调试：统计 sts_iter 被调用的次数
        for task_name in tqdm(task_names, desc=f'train-{epoch}', disable=TQDM_DISABLE):

        #先更新loss_components
            #if task_name=='sst':
            if last_task_name=='sts':
                try:
                   batch=next(sst_iter)
                except:
                    print(f"Epoch {epoch} error: sst_iter called {sst_count} times, sts_iter called {sts_count} times")
                    exit(1)
                sst_count += 1
                b_ids, b_mask, b_labels = (batch['token_ids'],
                                        batch['attention_mask'], batch['labels'])
                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)
                logits = model.predict_sentiment(b_ids, b_mask, task='sentiment')
                # 计算siar_penalty
                siar_penalty = compute_siar_penalty(model,
                                                    input_ids=b_ids,
                                                    attention_mask=b_mask,
                                                    input_ids_2=None,
                                                    attention_mask_2=None,
                                                    task='sentiment',
                                                    device=device,
                                                    epsilon=args.siar_eps,
                                                    labels=logits) if args.use_siar else torch.zeros(1, device=device, requires_grad=True).squeeze()
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size + siar_penalty*args.siar_coeff
                loss_components['sentiment'] = loss
                last_task_name='sst'
            # elif task_name=='para':  # 暂时注释掉para任务
            #     batch=next(para_iter)
            #     b_ids_1, b_mask_1, b_ids_2, b_mask_2,b_labels_1= (batch['token_ids_1'], batch['attention_mask_1'],
            #                                             batch['token_ids_2'], batch['attention_mask_2'],batch['labels'])
            #     b_labels_1 = b_labels_1.to(device).float()
            #     b_ids_1 = b_ids_1.to(device)
            #     b_mask_1 = b_mask_1.to(device)
            #     b_ids_2 = b_ids_2.to(device)
            #     b_mask_2 = b_mask_2.to(device)
            #     logitss = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
            #     # 计算siar_penalty
            #     siar_penalty = compute_siar_penalty(model,
            #                                         input_ids=b_ids_1,
            #                                         attention_mask=b_mask_1,
            #                                         input_ids_2=b_ids_2,
            #                                         attention_mask_2=b_mask_2,
            #                                         task='paraphrase',
            #                                         device=device,
            #                                         eps=args.siar_eps,
            #                                         steps=args.siar_steps,
            #                                         norm=args.siar_norm,
            #                                         epsilon=args.siar_eps,
            #                                         labels=logitss) if args.use_siar else torch.tensor(0.0, device=device, requires_grad=True)
            #     loss = F.binary_cross_entropy_with_logits(logitss, b_labels_1.view(-1).float(), reduction='sum') / args.batch_size + siar_penalty
            #     loss_components['paraphrase'] = loss
            #elif task_name=='sts':
            if last_task_name=='sst':
                try:
                   batch=next(sts_iter)
                except:
                    print(f"Epoch {epoch} error: sst_iter called {sst_count} times, sts_iter called {sts_count} times")
                    exit(1)
                sts_count += 1
                b_ids_3, b_mask_3, b_ids_4, b_mask_4,b_labels_2 = (batch['token_ids_1'], batch['attention_mask_1'],
                                                        batch['token_ids_2'], batch['attention_mask_2'],batch['labels'])
                b_labels_2 = b_labels_2.to(device).float()
                b_ids_3 = b_ids_3.to(device)
                b_mask_3 = b_mask_3.to(device)
                b_ids_4 = b_ids_4.to(device)
                b_mask_4 = b_mask_4.to(device)
                logitsss = model.predict_similarity(b_ids_3, b_mask_3, b_ids_4, b_mask_4)
                siar_penalty = compute_siar_penalty(model,
                                                    input_ids=b_ids_3,
                                                    attention_mask=b_mask_3,
                                                    input_ids_2=b_ids_4,
                                                    attention_mask_2=b_mask_4,
                                                    task='similarity',
                                                    device=device,
                                                    epsilon=args.siar_eps,
                                                    labels=logitsss) if args.use_siar else torch.tensor(0.0, device=device, requires_grad=True)
                loss = F.mse_loss(logitsss, b_labels_2.view(-1), reduction='sum') / args.batch_size + siar_penalty*args.siar_coeff
                loss_components['similarity'] = loss
                last_task_name='sts'
            
            
            optimizer.zero_grad()
            strategy = args.loss_strategy
            if last_task_name=='sts':#更新
                if strategy=='gradient_surgery':
                    grads=aggregate_task_losses(model, loss_components,strategy='gradient_surgery',weight_map_for_gradient_surgery=weight_map_for_gradient_surgery)
                    '''
                    grads已经被合并，只包含需要梯度的参数。
                    '''
                    trainable_params = [p for p in model.parameters() if p.requires_grad]
                    for p, g in zip(trainable_params, grads):
                        p.grad = g.detach().clone()
                    optimizer.step()
                    loss = sum(loss_components.values()) / len(loss_components)
                    loss_components.clear()
                
                elif strategy=='average':
                    loss=aggregate_task_losses(model, loss_components,strategy='average')
                    loss.backward()
                    optimizer.step()
                    
                elif strategy=='weighted':
                    # 解析权重映射
                    if args.loss_weights:
                        weight_map_for_weighted = {}
                        for item in args.loss_weights.split(','):
                            task, weight = item.split(':')
                            weight_map_for_weighted[task.strip()] = float(weight.strip())
                    else:
                        weight_map_for_weighted = {'sentiment': 1.0, 'similarity': 1.0}  # 暂时不包含paraphrase
                    loss=aggregate_task_losses(model, loss_components,strategy='weighted',weight_map_for_weighted=weight_map_for_weighted)
                    loss.backward()
                    optimizer.step()
                else:
                    # 默认使用average策略
                    loss=aggregate_task_losses(model, loss_components,strategy='average')
                    loss.backward()
                    optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # 定期清理显存缓存
            if global_step % 50 == 0:
                torch.cuda.empty_cache()
        train_loss=train_loss/ (num_batches)
                
        train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)
        train_acc_sts=model_eval_sts(sts_train_dataloader, model, device)
        dev_acc_sts=model_eval_sts(sts_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}:(sst)train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        print(f"Epoch {epoch}:(sts)train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")



def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="full-model")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--task_proj_dim", type=int, default=256,
                        help="任务投影层输出维度的默认值。")
    parser.add_argument("--paraphrase_proj_dim", type=int, default=None,
                        help="复述任务投影层输出维度（缺省时使用 task_proj_dim）。")
    parser.add_argument("--similarity_proj_dim", type=int, default=None,
                        help="相似度任务投影层输出维度（缺省时使用 task_proj_dim）。")

    parser.add_argument("--loss_strategy", type=str, default="gradient_surgery",
                        choices=("average", "weighted", "gradient_surgery"),
                        help="多任务损失聚合策略：average（平均）、weighted（加权）、gradient_surgery（梯度手术）。")
    parser.add_argument("--loss_weights", type=str, default=None,
                        help="自定义损失权重设置，格式如 sentiment:1.0,paraphrase:0.5。")
    parser.add_argument("--use_siar", type=bool, default=True,
                        help="启用平滑诱导对抗正则化（Smoothness-Inducing Adversarial Regularization）。默认启用。")
    parser.add_argument("--siar_eps", type=float, default=1e-1,
                        help="对抗扰动幅度 epsilon。")
    parser.add_argument("--siar_coeff", type=float, default=1.0,
                        help="正则项系数，用于控制SIAR损失对总体loss的影响。")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)

