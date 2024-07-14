import torch
from torch import nn
# from configs.paths_config import model_paths
import torchvision.models.vgg as vgg
import torch.nn.functional as F
# from losses.contextual_loss.functional import contextual_bilateral_loss


def compute_meshgrid(shape):
    N, C, H, W = shape
    rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
    cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

    feature_grid = torch.meshgrid(rows, cols)
    feature_grid = torch.stack(feature_grid).unsqueeze(0)
    feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

    return feature_grid


def compute_l2_distance(x, y):
    N, C, H, W = x.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)
    x_s = torch.sum(x_vec ** 2, dim=1, keepdim=True)
    y_s = torch.sum(y_vec ** 2, dim=1, keepdim=True)

    A = y_vec.transpose(1, 2) @ x_vec
    # print(x.shape, y_s.shape, A.shape, x_s.shape)
    dist = y_s - 2 * A + x_s.transpose(1, 2)
    dist = dist.transpose(1, 2).reshape(N, H*W, H*W)
    dist = dist.clamp(min=0.)

    return dist


def compute_relative_distance(dist_raw):
    dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
    dist_tilde = dist_raw / (dist_min + 1e-5)
    # add here.
    dist_tilde = torch.clamp(dist_tilde, max=10., min=-10)
    return dist_tilde


# def compute_cx(dist_tilde, band_width):
#     cx = torch.softmax((1 - dist_tilde) / band_width, dim=2)
#     return cx


def compute_cx(dist_tilde, band_width):
    # Easy to get NaN
    w = torch.exp((1 - dist_tilde) / band_width)  # Eq(3)
    cx = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)
    return cx


def compute_cosine_distance(x, y):
    # mean shifting by channel-wise mean of `y`.
    y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - y_mu
    y_centered = y - y_mu

    # L2 normalization
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)

    # channel-wise vectorization
    N, C, *_ = x.size()
    x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

    # consine similarity
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2),
                           y_normalized)  # (N, H*W, H*W)

    # convert to distance
    dist = 1 - cosine_sim

    return dist


def contextual_bilateral_loss(x: torch.Tensor,
                              y: torch.Tensor,
                              weight_sp: float = 0.1,
                              band_width: float = 1.,
                              loss_type: str = 'cosine'):
    """
    Computes Contextual Bilateral (CoBi) Loss between x and y,
        proposed in https://arxiv.org/pdf/1905.05169.pdf.

    Parameters
    ---
    x : torch.Tensor
        features of shape (N, C, H, W).
    y : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
        Note: `l1` and `l2` frequently raises OOM.

    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper).
    k_arg_max_NC : torch.Tensor
        indices to maximize similarity over channels.
    """

    # assert x.size() == y.size(), 'input tensor must have the same size.'
    # assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'

    # spatial loss
    grid = compute_meshgrid(x.shape).to(x.device)
    dist_raw = compute_l2_distance(grid, grid)
    dist_tilde = compute_relative_distance(dist_raw)
    cx_sp = compute_cx(dist_tilde, band_width)

    # feature loss
    # if loss_type == 'cosine':
    dist_raw = compute_cosine_distance(x, y)
    
    dist_tilde = compute_relative_distance(dist_raw)
    cx_feat = compute_cx(dist_tilde, band_width)

    # combined loss
    cx_combine = (1. - weight_sp) * cx_feat + weight_sp * cx_sp

    k_max_NC, _ = torch.max(cx_combine, dim=2, keepdim=True)

    cx = k_max_NC.mean(dim=1)
    cx_loss = torch.mean(-torch.log(cx + 1e-5))

    return cx_loss


def contextual_loss(x: torch.Tensor,
                    y: torch.Tensor,
                    band_width: float = 0.5,):
                    # loss_type: str = 'cosine'):
    """
    Computes contextual loss between x and y.
    The most of this code is copied from
        https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da.

    Parameters
    ---
    x : torch.Tensor
        features of shape (N, C, H, W).
    y : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
        Note: `l1` and `l2` frequently raises OOM.

    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper)
    """
    # assert x.size() == y.size(), 'input tensor must have the same size.'
    # assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'

    N, C, H, W = x.size()

    dist_raw = compute_cosine_distance(x, y)

    dist_tilde = compute_relative_distance(dist_raw)
    cx = compute_cx(dist_tilde, band_width)
    # torch.sum(compute_cx_bak(dist_tilde, band_width) == cx)
    cx = torch.mean(torch.max(cx, dim=1)[0], dim=1)  # Eq(1)
    cx_loss = torch.mean(-torch.log(cx + 1e-5))  # Eq(5)

    return cx_loss



class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = vgg.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        for x in range(18):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
      
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        return h


class CXLoss(nn.Module):
    def __init__(self,
                 band_width: float = 0.5):
        super(CXLoss, self).__init__()
        self.band_width = band_width

        self.vgg_model = VGG19()
        # self.register_buffer(
        #     name='vgg_mean',
        #     tensor=torch.tensor(
        #         [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
        # )
        # self.register_buffer(
        #     name='vgg_std',
        #     tensor=torch.tensor(
        #         [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
        # )

    def forward(self, x, y):
        if x.shape[-1] > 256:
            x = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
            y = F.interpolate(y, (256, 256), mode='bilinear', align_corners=False)
        # # normalization
        # x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
        # y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())

        # # picking up vgg feature maps
        # x = getattr(self.vgg_model(x), self.vgg_layer)
        # y = getattr(self.vgg_model(y), self.vgg_layer)
        x = self.vgg_model(x)
        y = self.vgg_model(y)
        loss = contextual_loss(x, y, self.band_width)
        return loss
        # return contextual_bilateral_loss(x, y, self.band_width)