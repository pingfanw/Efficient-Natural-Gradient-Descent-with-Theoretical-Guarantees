import math
import torch
import torch.optim as optim
import torch.linalg as LA
from utils.kfac_utils import (ComputeCovA, ComputeCovG)
from utils.kfac_utils import update_running_stat
from utils.kfac_utils import ComputeMatGrad
from torch import Tensor
from torch.optim.optimizer import (Optimizer, required, _use_grad_for_differentiable, _default_to_fused_or_foreach,
                        _differentiable_doc, _foreach_doc, _maximize_doc)
from typing import List, Optional
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

class SGD_mod(optim.Optimizer):
    def __init__(self,
                 params,
                 lr=0.01,
                 momentum=0.9,
                 weight_decay=0):
        # legitimation check
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, params=params,
                        weight_decay=weight_decay)
        super(SGD_mod, self).__init__(params, defaults)
    def _step(self, closure):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            fisher = 0
            for param in group['params']:
                if param.grad is None:
                    continue
                d_p = (param.grad.data)
                if weight_decay != 0:
                    d_p.add_(param.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[param]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(param.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1)
                    d_p = buf
                param.data.add_(d_p, alpha=-group['lr'])
    def step(self, closure=None):
        self._step(closure)


class Adam_mod(optim.Optimizer):
    def __init__(self,
                 params,
                 lr=0.001,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if betas[0] < 0.0 or betas[0] >= 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if betas[1] < 0.0 or betas[1] >= 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam_mod, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_mod, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam_mod does not support sparse gradients, please consider SparseAdam instead')
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if group['amsgrad']:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (bias_correction2**(1/2) / (bias_correction1))#.clamp(min=1e-6)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        return loss


class DNGD(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.01,
                 momentum=0.9,
                 weight_decay=0,
                 damping=1e-1):
        # legitimation check
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, damping=damping)
        super(DNGD, self).__init__(model.parameters(), defaults)
        self.known_modules = {'Linear', 'Conv2d'}
        self.modules = []
        """ list for saving modules temporarily """
        self.model = model
        # self.damping = damping
        self._prepare_model()          

    def _prepare_model(self):
        # print(self.model)
        for module in self.model.modules():      
            classname = module.__class__.__name__       
            if classname in self.known_modules:    
                self.modules.append(module)        

    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the mth layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            param_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1)  # n_filters * (in_c * kw * kh) 
        else:
            param_grad_mat = m.weight.grad.data
        if m.bias is not None:
            param_grad_mat = torch.cat([param_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return param_grad_mat

    def _get_natural_grad(self, m, param_grad_mat, damping):
        """
        :param m:  the mth layer
        :param param_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m` th layer
        """
        v = param_grad_mat
        v *= torch.reciprocal(torch.square(param_grad_mat).sum(dim=0,keepdim=True)+damping)

        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]
        return v

    def _update_grad(self, updates):        
        # update grad with fvp
        for m in self.modules:
            v = updates[m]
            m.weight.grad.data.copy_(v[0])      # update the weight, replacing the original gradient with F^-1*nebla(h)
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])        # update the bias

    def _step(self, closure):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            for param in group['params']:
                if param.grad is None:
                    continue
                d_p = param.grad.data
                if weight_decay != 0:
                    d_p.add_(param.data, alpha=weight_decay)
                if momentum != 0:                                       # add momentum
                    param_state = self.state[param]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(param.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1)
                    d_p = buf
                param.data.add_(d_p, alpha=-group['lr'])      # update the parameters

    def step(self, closure=None):
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        for m in self.modules:
            classname = m.__class__.__name__                                 #update the inverse of the fisher by implementing eigen decomposition of kronecker factor
            param_grad_mat = self._get_matrix_form_grad(m, classname)       #acquiring the grad matrix of m layer (grad_mat=cat[weight,bias])
            fvp = self._get_natural_grad(m, param_grad_mat, damping)          # acquiring the fisher vector product if m layer
            updates[m] = fvp                                               # put the fvp of m layer about bias and weight into updates[m]
        self._update_grad(updates)                         
        self._step(closure)         #update the parameters


class KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=5,
                 TInv=50,
                 batch_averaged=True):
        # legitimation check
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        # TODO (CW): KFAC optimizer now only support model as input
        super(KFACOptimizer, self).__init__(model.parameters(), defaults)

        self.CovAHandler = ComputeCovA()
        """ compute the covariance of the activation """
        self.CovGHandler = ComputeCovG()
        """ compute the covariance of the gradient """
        self.batch_averaged = batch_averaged
        """ bool markers for whether the gradient is batch averaged """
        
        self.known_modules = {'Linear', 'Conv2d'}
        """ dictionary for modules: {Linear,Conv2d} """

        self.modules = []
        """ list for saving modules temporarily """

        self.grad_outputs = {}
        """ buffer for saving the gradient output """
        self.model = model
        self._prepare_model()          

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        """ buffer for saving the running estimates of the covariance of the activation and gradient """
        self.Q_a, self.Q_g = {}, {}
        """ buffer for saving the eigenvectors of the covariance of the activation and gradient """
        self.d_a, self.d_g = {}, {}
        """ buffer for saving the eigenvalues of the covariance of the activation and gradient """
        self.stat_decay = stat_decay
        """ parameter determines the time scale for the moving average """

        self.kl_clip = kl_clip
        self.TCov = TCov
        """ the period for computing the covariance of the activation and gradient """
        self.TInv = TInv
        """ the period for updating the inverse of the covariance """

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            aa = self.CovAHandler(input[0].data, module)   
            # print("aa size:",aa.size())     
            # Initialize buffers
            if self.steps == 0:     
                self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
                # print("m_aa size:",self.m_aa[module].size())
            update_running_stat(aa, self.m_aa[module], self.stat_decay)             # exponential moving average

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.TCov == 0:
            gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
            update_running_stat(gg, self.m_gg[module], self.stat_decay)             # exponential moving average
    
    def _prepare_model(self):
        count = 0
        # print(self.model)
        # print("=> We keep following layers in KFAC. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)  
                module.register_full_backward_hook(self._save_grad_output)
                # print('=>(%s): %s' % (count, module))
                count += 1

    def _update_inv(self, m):
        """
        Do eigen decomposition of kronecker faction for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        :function: 
            m_aa=Q_a*(d_a)*Q_a^T
            m_gg=Q_g*(d_g)*Q_g^T
        """
        eps = 1e-10  # for numerical stability
        self.d_a[m], self.Q_a[m] = LA.eigh(
            self.m_aa[m], UPLO='L')
        # print(self.Q_a[m])
        # print("Q_a size:",self.Q_a[m].size())
        self.d_g[m], self.Q_g[m] = LA.eigh(
            self.m_gg[m], UPLO='L')
        self.d_a[m].mul_((self.d_a[m] > eps).float())
        self.d_g[m].mul_((self.d_g[m] > eps).float())

    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the mth layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            param_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1)
            # print("param_grad_mat size:",param_grad_mat.size())
        else:
            param_grad_mat = m.weight.grad.data
        if m.bias is not None:
            param_grad_mat = torch.cat([param_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return param_grad_mat

    def _get_natural_grad(self, m, param_grad_mat, damping):
        """
        :param m:  the mth layer
        :param param_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m` th layer
        """
        v1 = self.Q_g[m].t() @ param_grad_mat @ self.Q_a[m]
        v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
        v = self.Q_g[m] @ v2 @ self.Q_a[m].t()                                      
        if m.bias is not None:
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]
            
        return v

    def _kl_clip_and_update_grad(self, updates, lr):        #paper conv-kfac appendix A.1
        # do kl clip
        vg_sum = 0
        for m in self.modules:
            v = updates[m]
            vg_sum += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
            if m.bias is not None:
                vg_sum += (v[1] * m.bias.grad.data * lr ** 2).sum().item()
        nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))
        # update grad with fvp
        for m in self.modules:
            v = updates[m]
            m.weight.grad.data.copy_(v[0])      # update the weight, replacing the original gradient with F^-1*nebla(h)
            m.weight.grad.data.mul_(nu)     # multiply the kl_clip factor
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])        # update the bias
                m.bias.grad.data.mul_(nu)       # multiply the kl_clip factor

    def _step(self, closure):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            for param in group['params']:
                if param.grad is None:
                    continue
                d_p = param.grad.data
                if weight_decay != 0 and self.steps >= 20 * self.TCov:          # do regularization after 20 TCov
                    d_p.add_(param.data, alpha=weight_decay)
                if momentum != 0:                                       # add momentum
                    param_state = self.state[param]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(param.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1)
                    d_p = buf

                param.data.add_(d_p, alpha=-group['lr'])      # update the parameters

    def step(self, closure=None):
        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        for m in self.modules:
            classname = m.__class__.__name__
            if self.steps % self.TInv == 0:
                self._update_inv(m)                                         #update the inverse of the fisher by implementing eigen decomposition of kronecker factor
            param_grad_mat = self._get_matrix_form_grad(m, classname)       #acquiring the grad matrix of m layer (grad_mat=cat[weight,bias])
            fvp = self._get_natural_grad(m, param_grad_mat, damping)          # acquiring the fisher vector product if m layer
            updates[m] = fvp                                               # put the fvp of m layer about bias and weight into updates[m]
        self._kl_clip_and_update_grad(updates, lr)                          #do kl clip and update grad

        self._step(closure)         #update the parameters
        self.steps += 1


class KFACOptimizer_NonFull(KFACOptimizer):
    def _prepare_model(self):
        count = 0
        # print(self.model)
        print("=> We keep following layers in KFAC. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)  
                module.register_backward_hook(self._save_grad_output)
                count += 1


class EKFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=10,
                 TScal=10,
                 TInv=100,
                 batch_averaged=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        # TODO (CW): EKFAC optimizer now only support model as input
        super(EKFACOptimizer, self).__init__(model.parameters(), defaults)
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.MatGradHandler = ComputeMatGrad()
        self.batch_averaged = batch_averaged

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.S_l = {}
        self.A, self.DS = {}, {}
        self.stat_decay = stat_decay

        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TScal = TScal
        self.TInv = TInv

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            aa = self.CovAHandler(input[0].data, module)
            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
            update_running_stat(aa, self.m_aa[module], self.stat_decay)
        if torch.is_grad_enabled() and self.steps % self.TScal == 0  and self.steps > 0:
            self.A[module] = input[0].data

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.TCov == 0:
            gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
            update_running_stat(gg, self.m_gg[module], self.stat_decay)

        # if self.steps % self.TInv == 0:
        #     self._update_inv(module)

        if self.acc_stats and self.steps % self.TScal == 0 and self.steps > 0:
            self.DS[module] = grad_output[0].data
            # self._update_scale(module)

    def _prepare_model(self):
        count = 0
        # print(self.model)
        print("=> We keep following layers in EKFAC. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)
                print('(%s): %s' % (count, module))
                count += 1

    def _update_inv(self, m):
        """Do eigen decomposition for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
        eps = 1e-10  # for numerical stability
        self.d_a[m], self.Q_a[m] = LA.eigh(
            self.m_aa[m], UPLO='L')
        self.d_g[m], self.Q_g[m] = LA.eigh(
            self.m_gg[m], UPLO='L')

        self.d_a[m].mul_((self.d_a[m] > eps).float())
        self.d_g[m].mul_((self.d_g[m] > eps).float())
        # if self.steps != 0:
        self.S_l[m] = self.d_g[m].unsqueeze(1) @ self.d_a[m].unsqueeze(0)

    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            p_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1)  # n_filters * (in_c * kw * kh)
        else:
            p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def _get_natural_grad(self, m, p_grad_mat, damping):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # p_grad_mat is of output_dim * input_dim
        # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]
        v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
        v2 = v1 / (self.S_l[m] + damping)
        v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]

        return v

    def _kl_clip_and_update_grad(self, updates, lr):
        # do kl clip
        vg_sum = 0
        for m in self.modules:
            v = updates[m]
            vg_sum += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
            if m.bias is not None:
                vg_sum += (v[1] * m.bias.grad.data * lr ** 2).sum().item()
        nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))

        for m in self.modules:
            v = updates[m]
            m.weight.grad.data.copy_(v[0])
            m.weight.grad.data.mul_(nu)
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
                m.bias.grad.data.mul_(nu)

    def _step(self, closure):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0 and self.steps >= 20 * self.TCov:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1)
                    d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])

    def _update_scale(self, m):
        with torch.no_grad():
            A, S = self.A[m], self.DS[m]
            grad_mat = self.MatGradHandler(A, S, m)  # batch_size * out_dim * in_dim
            if self.batch_averaged:
                grad_mat *= S.size(0)

            s_l = (self.Q_g[m] @ grad_mat @ self.Q_a[m].t()) ** 2  # <- this consumes too much memory!
            s_l = s_l.mean(dim=0)
            if self.steps == 0:
                self.S_l[m] = s_l.new(s_l.size()).fill_(1)
            update_running_stat(s_l, self.S_l[m], self.stat_decay)
            # remove reference for reducing memory cost.
            self.A[m] = None
            self.DS[m] = None

    def step(self, closure=None):
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        for m in self.modules:
            classname = m.__class__.__name__
            if self.steps % self.TInv == 0:
                self._update_inv(m)

            if self.steps % self.TScal == 0 and self.steps > 0:
                self._update_scale(m)

            p_grad_mat = self._get_matrix_form_grad(m, classname)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v
        self._kl_clip_and_update_grad(updates, lr)

        self._step(closure)
        self.steps += 1

class EKFACOptimizer_NonFull(EKFACOptimizer):
    def _prepare_model(self):
        count = 0
        # print(self.model)
        print("=> We keep following layers in EKFAC. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                print('(%s): %s' % (count, module))
                count += 1
