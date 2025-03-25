import torch
import torch.nn as nn
import torch.nn.functional as F


def try_contiguous(x):
    """
    test x is contiguousor not, 
    for acquiring the covariance of gradient execlusively in conv2d layer.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    return x


def _extract_patches(x, kernel_size, stride, padding):
    """
    :param x: The input feature maps. 
        (batch_size, in_channel, h, w) 
    :param kernel_size: the kernel size of the conv 2d filter 
        (tuple of two elements ie.[kernel_h, kernel_w])
    :param stride: the stride of conv 2d operation. 
        (tuple of two elements ie.[stride_h, stride_w])
    :param padding: number of paddings. 
        (tuple of two elements ie.[padding_h, padding_w])
    :return: extracted batches for the input feature maps of corresponding conv2d layer.
        (batch_size, out_h, out_w, in_c*kernel_h*kernel_w)
    """
    if padding[0] + padding[1] > 0:     
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data       
    x = x.unfold(2, kernel_size[0], stride[0])      
    x = x.unfold(3, kernel_size[1], stride[1])        
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()           
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))   
    return x


def update_running_stat(aa, m_aa, stat_decay):
    """ 
    update the running estimates for the covariance of activation or gradient.
        ie. exponential moving average in the end of 3.1 in the paper.
    
    :param aa: the covariance of activation or gradient.
    :param m_aa: the updated running estimates of covariance of activation or gradient.
    :param stat_decay: the parameter determines the time scale for the moving average.
    :return: None
    :Function: m_aa = stat_decay*m_aa + (1-stat_decay)*aa
    """
    # using inplace operation to save memory!
    m_aa *= stat_decay / (1 - stat_decay)
    m_aa += aa
    m_aa *= (1 - stat_decay)

class ComputeCovA:
    @classmethod
    def compute_cov_a(cls, a, layer):
        """
        :param a: intermediates' activation
        :param layer: the corresponding layer
        :return:__call__ returns the covariance of activation
        """
        return cls.__call__(a, layer)
    
    @classmethod
    def __call__(cls, a, layer):
        if isinstance(layer, nn.Conv2d):
            cov_a = cls.cova_conv2d(a, layer)
        elif isinstance(layer, nn.Linear):
            cov_a = cls.cova_linear(a, layer)
        else:
            cov_a = None
        return cov_a

    @staticmethod
    def cova_conv2d(a, layer):
        """ compute the covariance of activation in conv2d layer. """
        batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        spatial_size = a.size(1) * a.size(2)   
        a = a.view(-1, a.size(-1))
        if layer.bias is not None:    
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a/spatial_size
        return a.t() @ (a / batch_size)  

    @staticmethod
    def cova_linear(a, layer):
        """ compute the covariance of activation in linear layer. """
        batch_size = a.size(0)
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], dim=1)
        return a.t() @ (a / batch_size)

class ComputeCovG:
    @classmethod
    def compute_cov_g(cls, g, layer, batch_averaged=False):
        """
        :param g: gradient
        :param layer: the corresponding layer
        :param batch_averaged: if the gradient is already averaged with the batch size?
        :return:__call__ returns the covariance of gradient
        """
        return cls.__call__(g, layer, batch_averaged)

    @classmethod
    def __call__(cls, g, layer, batch_averaged):
        if isinstance(layer, nn.Conv2d):
            cov_g = cls.covg_conv2d(g, layer, batch_averaged)
        elif isinstance(layer, nn.Linear):
            cov_g = cls.covg_linear(g, layer, batch_averaged)
        else:
            cov_g = None

        return cov_g

    @staticmethod
    def covg_conv2d(g, layer, batch_averaged):
        """ compute the covariance of gradient in conv2d layer. """
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0] 
        g = g.transpose(1, 2).transpose(2, 3)       
        g = try_contiguous(g)       #continuous g after transpose
        g = g.view(-1, g.size(-1))
        if batch_averaged:
            g = g * batch_size      #cancel batch_averaged
        g = g * spatial_size
        cov_g = g.t() @ (g / g.size(0))

        return cov_g

    @staticmethod
    def covg_linear(g, layer, batch_averaged):
        """ compute the covariance of gradient in linear layer. """
        batch_size = g.size(0)
        if batch_averaged:
            cov_g = g.t() @ (g * batch_size)        #cancel batch_averaged
        else:
            cov_g = g.t() @ (g / batch_size)
        return cov_g

class ComputeMatGrad:
    """ ongly for EKFAC """
    @classmethod
    def __call__(cls, input, grad_output, layer):
        if isinstance(layer, nn.Linear):
            grad = cls.linear(input, grad_output, layer)
        elif isinstance(layer, nn.Conv2d):
            grad = cls.conv2d(input, grad_output, layer)
        else:
            raise NotImplementedError
        return grad

    @staticmethod
    def linear(input, grad_output, layer):
        """
        :param input: batch_size * input_dim
        :param grad_output: batch_size * output_dim
        :param layer: [nn.module] output_dim * input_dim
        :return: batch_size * output_dim * (input_dim + [1 if with bias])
        """
        with torch.no_grad():
            if layer.bias is not None:
                input = torch.cat([input, input.new(input.size(0), 1).fill_(1)], 1)
            input = input.unsqueeze(1)
            grad_output = grad_output.unsqueeze(2)
            grad = torch.bmm(grad_output, input)
        return grad

    @staticmethod
    def conv2d(input, grad_output, layer):
        """
        :param input: batch_size * in_c * in_h * in_w
        :param grad_output: batch_size * out_c * h * w
        :param layer: nn.module batch_size * out_c * (in_c*k_h*k_w + [1 if with bias])
        :return:
        """
        with torch.no_grad():
            input = _extract_patches(input, layer.kernel_size, layer.stride, layer.padding)
            input = input.view(-1, input.size(-1))  # b * hw * in_c*kh*kw
            grad_output = grad_output.transpose(1, 2).transpose(2, 3)
            grad_output = try_contiguous(grad_output).view(grad_output.size(0), -1, grad_output.size(-1))
            # b * hw * out_c
            if layer.bias is not None:
                input = torch.cat([input, input.new(input.size(0), 1).fill_(1)], 1)
            input = input.view(grad_output.size(0), -1, input.size(-1))  # b * hw * in_c*kh*kw
            grad = torch.einsum('abm,abn->amn', (grad_output, input))
        return grad
    

if __name__ == '__main__':
    def test_ComputeCovA():
        pass

    def test_ComputeCovG():
        pass
