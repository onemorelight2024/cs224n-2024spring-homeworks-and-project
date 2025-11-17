from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:#param_group每个组对应一个模块，如卷积层、全连接层
            for p in group["params"]:#p是一个模块里的每个矩阵
                if p.grad is None:
                    continue
                grad = p.grad.data#优化器里要用p.data/p.grad.data防止根据梯度修改原参数
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State里储藏着该矩阵的所有信息
                state = self.state[p]
                
                
                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]
                beta1,beta2=group["betas"]
                eps=group["eps"]
                
                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # To complete this implementation:
                # 1. Update the first and second moments of the gradients.
                # 2. Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3. Update parameters (p.data).
                # 4. Apply weight decay after the main gradient-based updates.
                # Refer to the default project handout for more details.
                
                ### TODO
                if len(state)==0:
                    state["t"]=0
                    state["m_t"]=torch.zeros_like(p.data)
                    state["v_t"]=torch.zeros_like(p.data)
                state["t"]+=1
                m_t_pre=state["m_t"]
                state["m_t"]=(1-beta1)*grad+beta1*m_t_pre
                v_t_pre=state["v_t"]
                state["v_t"]=(1-beta2)*grad*grad+beta2*v_t_pre
                
                alpha_t=alpha*math.sqrt(1-beta2**state["t"])/(1-beta1**state["t"])#有t次方，保证刚开始的修正
                #alpha_t是t步的学习率，而不是alpha
                
                p.data=p.data-alpha_t*state["m_t"] /(torch.sqrt(state["v_t"]) + eps)
                
                #应用AdamW的权重衰减 4. Apply weight decay after the main gradient-based updates.
                p.data = p.data - group["lr"] * group["weight_decay"] * p.data
                #参数组的weight_decay而不是self.weight_dacay

        return loss