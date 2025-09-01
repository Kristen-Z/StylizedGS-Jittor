# import jt
import jittor as jt
from jittor import nn
# import torchvision
# from torchvision.models import vgg16, VGG16_Weights
from icecream import ic
# import jt.nn.functional as F
from typing import Tuple
from jittor.models import vgg16
from jittor.transform import ImageNormalize



def match_colors_for_image_set(image_set, style_img):
    """
    image_set: [N, H, W, 3]
    style_img: [H, W, 3]
    """
    sh = image_set.shape
    image_set = image_set.view(-1, 3)
    style_img = style_img.view(-1, 3)

    mu_c = image_set.mean(0, keepdim=True)
    mu_s = style_img.mean(0, keepdim=True)

    cov_c = jt.matmul((image_set - mu_c).transpose(1, 0), image_set - mu_c) / float(image_set.size(0))
    cov_s = jt.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s) / float(style_img.size(0))

    u_c, sig_c, _ = jt.linalg.svd(cov_c)

    u_s, sig_s, _ = jt.linalg.svd(cov_s)

    u_c_i = u_c.transpose(1, 0)
    u_s_i = u_s.transpose(1, 0)

    scl_c = jt.diag(1.0 / jt.sqrt(jt.clamp(sig_c, 1e-8, 1e8)))
    scl_s = jt.diag(jt.sqrt(jt.clamp(sig_s, 1e-8, 1e8)))

    tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i

    tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.transpose(0,1)

    image_set = image_set @ tmp_mat.transpose(0,1) + tmp_vec.view(1, 3)
    image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)

    color_tf = jt.init.eye(4).float()
    color_tf[:3, :3] = tmp_mat

    color_tf[:3, 3:4] = tmp_vec.transpose(0,1)
    return image_set, color_tf

def color_histgram_match(
    input: jt.Var,
    source: jt.Var,
    mode: str = "pca",
    eps: float = 1e-5,
) -> jt.Var:
    """
    Transfer the colors from one image Var to another, so that the target image's
    histogram matches the source image's histogram. Applications for image histogram
    matching includes neural style transfer and astronomy.

    The source image is not required to have the same height and width as the target
    image. Batch and channel dimensions are required to be the same for both inputs.

    Gatys, et al., "Controlling Perceptual Factors in Neural Style Transfer", arXiv, 2017.
    https://arxiv.org/abs/1611.07865

    Args:

        input (jt.Var): The NCHW or CHW image to transfer colors from source
            image to from the source image.
        source (jt.Var): The NCHW or CHW image to transfer colors from to the
            input image.
        mode (str): The color transfer mode to use. One of 'pca', 'cholesky', or 'sym'.
            Default: "pca"
        eps (float): The desired epsilon value to use.
            Default: 1e-5

    Returns:
        matched_image (jt.Var): The NCHW input image with the colors of source
            image. Outputs should ideally be clamped to the desired value range to
            avoid artifacts.
    """

    sh = input.shape
    input = input.view(-1, 3)
    source = source.view(-1, 3)

    # print('histogram match', source.shape, input.shape)

    # Handle older versions of PyTorch
    # torch_cholesky = (
    #     jt.linalg.cholesky if jt.__version__ >= "1.9.0" else jt.cholesky
    # )
    jittor_cholesky = (jt.linalg.cholesky)

    # def torch_symeig_eigh(x: jt.Var) -> Tuple[jt.Var, jt.Var]:
    #     """
    #     jt.symeig() was deprecated in favor of jt.linalg.eigh()
    #     """
    #     if jt.__version__ >= "1.9.0":
    #         L, V = jt.linalg.eigh(x, UPLO="U")
    #     else:
    #         L, V = jt.symeig(x, eigenvectors=True, upper=True)
    #     return L, V
    def jittor_symeig_eigh(x: jt.Var) -> Tuple[jt.Var, jt.Var]:
        """
        jt.symeig() was deprecated in favor of jt.linalg.eigh()
        """
        x = x.triu()
        L, V = jt.linalg.eigh(x + x.triu(1).transpose(1,2))
        return L, V
    
    def get_mean_vec_and_cov(
        x_input: jt.Var, eps: float
    ) -> Tuple[jt.Var, jt.Var, jt.Var]:
        """
        Convert input images into a vector, subtract the mean, and calculate the
        covariance matrix of colors.
        """
        x_mean = x_input.mean(0, keepdim=True)

        x_vec = x_input - x_mean
        x_cov = jt.matmul(x_vec.transpose(1, 0), x_vec) / float(x_input.size(0))

        # This line is only important if you get artifacts in the output image
        x_cov = x_cov + (eps * jt.init.eye(x_input.size(-1))[None, :])
        return x_mean, x_vec, x_cov
    def diag_embed_jittor(eigenvalues):
        B, D = eigenvalues.shape
        eye = jt.init.eye(D).unsqueeze(0)         # (1, D, D)
        eye = eye.broadcast((B, D, D))       # (B, D, D)
        diag_matrix = eye * eigenvalues.unsqueeze(-1)  # (B, D, D)
        return diag_matrix
    def pca(x: jt.Var) -> jt.Var:
        """Perform principal component analysis"""

        eigenvalues, eigenvectors = jittor_symeig_eigh(x)
        e = jt.sqrt(diag_embed_jittor(eigenvalues.reshape(eigenvalues.size(0), -1)))

        # Remove any NaN values if they occur
        if jt.isnan(e).any():
            e = jt.where(jt.isnan(e), jt.zeros_like(e), e)
        return jt.bmm(jt.bmm(eigenvectors, e), eigenvectors.permute(0, 2, 1))

    # Collect & calculate required values

    _, input_vec, input_cov = get_mean_vec_and_cov(input, eps)
    source_mean, _, source_cov = get_mean_vec_and_cov(source, eps)
    # Calculate new cov matrix for input
    if mode == "pca":
        new_cov = jt.bmm(pca(source_cov), jt.linalg.inv(pca(input_cov)))
    elif mode == "cholesky":
        new_cov = jt.bmm(
            jittor_cholesky(source_cov), jt.linalg.inv(jittor_cholesky(input_cov))
        )
    elif mode == "sym":
        p = pca(input_cov)
        pca_out = pca(jt.bmm(jt.bmm(p, source_cov), p))
        new_cov = jt.bmm(jt.bmm(jt.linalg.inv(p), pca_out), jt.linalg.inv(p))
    else:
        raise ValueError(
            "mode has to be one of 'pca', 'cholesky', or 'sym'."
            + " Received '{}'.".format(mode)
        )

    # Multiply input vector by new cov matrix
    # print('new_vec', new_cov.shape)

    new_vec = input_vec @ new_cov[0].transpose(0,1)

    # Reshape output vector back to input's shape &
    # add the source mean to our output vector
    image_set = (new_vec.reshape(sh) + source_mean).clamp_(0.0, 1.0)

    # print('new cov', new_cov.shape, 'source_mean', source_mean.shape)
    
    color_tf = jt.init.eye(4).float()
    color_tf[:3, :3] = new_cov[0]
    color_tf[:3, 3:4] = source_mean.transpose(0,1)
    return image_set, color_tf


def argmin_cos_distance(a, b, center=False):
    """
    a: [b, c, hw],
    b: [b, c, h2w2]
    """
    if center:
        a = a - a.mean(2, keepdims=True)
        b = b - b.mean(2, keepdims=True)

    b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()
    b = b / (b_norm + 1e-8)

    z_best = []
    loop_batch_size = int(1e8 / b.shape[-1])
    for i in range(0, a.shape[-1], loop_batch_size):
        a_batch = a[..., i : i + loop_batch_size]
        a_batch_norm = ((a_batch * a_batch).sum(1, keepdims=True) + 1e-8).sqrt()
        a_batch = a_batch / (a_batch_norm + 1e-8)

        d_mat = 1.0 - jt.matmul(a_batch.transpose(2, 1), b)

        z_best_batch, _ = jt.argmin(d_mat, 2)
        # print(z_best_batch)
        z_best.append(z_best_batch)
    z_best = jt.cat(z_best, dim=-1)

    return z_best


def nn_feat_replace(a, b):
    n, c, h, w = a.size()
    n2, c, h2, w2 = b.size()

    assert (n == 1) and (n2 == 1)

    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c, -1)
    b_ref = b_flat.clone()

    z_new = []
    for i in range(n):
        z_best = argmin_cos_distance(a_flat[i : i + 1], b_flat[i : i + 1])
        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
        feat = jt.gather(b_ref, 2, z_best)
        z_new.append(feat)

    z_new = jt.cat(z_new, 0)
    z_new = z_new.view(n, c, h, w)
    return z_new


def cos_loss(a, b):
    a_norm = (a * a).sum(1, keepdims=True).sqrt()
    
    b_norm = (b * b).sum(1, keepdims=True).sqrt()
    a_tmp = a / (a_norm + 1e-8)
    b_tmp = b / (b_norm + 1e-8)
    cossim = (a_tmp * b_tmp).sum(1)
    cos_d = 1.0 - cossim
    return cos_d.mean()


def gram_matrix(feature_maps, center=False):
    """
    feature_maps: b, c, h, w
    gram_matrix: b, c, c
    """
    b, c, h, w = feature_maps.size()
    features = feature_maps.view(b, c, h * w)
    if center:
        features = features - features.mean(dim=-1, keepdims=True)
    G = jt.bmm(features, jt.transpose(features, 1, 2))
    return G

def lum_transform(img):
    """
    Returns the projection of a colour image onto the luminance channel
    Images are expected to be of form (w,h,c) and float in [0,1].
    """
    lum = img[:,:1,:,:]*0.299 + img[:,1:2,:,:]*0.587 + img[:,2:,:,:]*0.114
    return lum.repeat(1,3,1,1)

def content_loss(feat_result, feat_content):
    d = feat_result.size(1)

    X = feat_result.transpose(0,1).contiguous().view(d,-1).transpose(0,1)
    Y = feat_content.transpose(0,1).contiguous().view(d,-1).transpose(0,1)

    Y = Y[:,:-2]
    X = X[:,:-2]
    # X = X.t()
    # Y = Y.t()

    Mx = cos_loss(X, X)
    Mx = Mx#/Mx.sum(0, keepdim=True)

    My = cos_loss(Y, Y)
    My = My#/My.sum(0, keepdim=True)

    d = jt.abs(Mx-My).mean()# * X.shape[0]
    return d

def crop_nonzero_region(image, padding=5):
    # 找到非零元素的坐标
    nonzero_indices = jt.nonzero(image != 0)

    row, col = image.shape[-2:]
    if len(nonzero_indices) == 0:
        # 如果图像中没有非零元素，返回空张量或其他适当的处理
        return jt.Var([])

    # 计算最小包围矩形的坐标范围
    min_row = jt.min(nonzero_indices[:, 2])
    max_row = jt.max(nonzero_indices[:, 2])
    min_col = jt.min(nonzero_indices[:, 3])
    max_col = jt.max(nonzero_indices[:, 3])


    # 切片原始图像以获得矩形区域
    cropped_image = image[:, :, max(0, min_row-padding):min(row, max_row + 1+padding), max(0, min_col-padding):min(col, max_col + 1+padding)]

    # print('cropped feat', cropped_image.shape)
    return cropped_image


class NNFMLoss(jt.nn.Module):
    def __init__(self, device):
        super().__init__()

        # self.vgg = vgg16(weights=VGG16_Weights.DEFAULT).eval()
        # self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.vgg = vgg16(pretrained=True).eval()
        for param in self.vgg.parameters():
            param.stop_grad()
        self.normalize = ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def get_feats(self, x, layers=[]):
        x = self.normalize(x)
        final_ix = max(layers)
        outputs = []

        for ix, layer in enumerate(self.vgg.features):
            x = layer(x)
            if ix in layers:
                outputs.append(x)

            if ix == final_ix:
                break

        return outputs

    def execute(
        self,
        outputs,
        styles,
        blocks=[
            2,
        ],
        loss_names=["nnfm_loss"],  # can also include 'gram_loss', 'content_loss'
        contents=None,
        layer_coef=[1.0,1.0],
        x_mask=None,
        s_mask=None,
        styles2=None,
    ):
        
        # styles = styles.stop_grad()  # 阻断 styles 的梯度
        # if contents is not None:
        #     contents = contents.stop_grad()  # 阻断 contents 的梯度

        for x in loss_names:
            assert x in ['nnfm_loss', 'content_loss', 'gram_loss', 'lum_nnfm_loss', "spatial_loss", "scale_loss"]

        block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]

        blocks.sort()
        all_layers = []
        for block in blocks:
            all_layers += block_indexes[block]
        # print('mask', x_mask.shape, outputs.shape,x_mask)
        if x_mask is not None:
            x_feats_all = self.get_feats(outputs*(1-x_mask), all_layers)
        else:
            x_feats_all = self.get_feats(outputs, all_layers)
        # ic(x_feats_all[0][0])
        with jt.no_grad():
            s_feats_all = self.get_feats(styles, all_layers)
            if "content_loss" in loss_names:
                content_feats_all = self.get_feats(contents, all_layers)
        
        # For spatial control
        if "spatial_loss" in loss_names:
            masked_x = crop_nonzero_region(outputs * x_mask)
            x_feats_mask = self.get_feats(masked_x, all_layers)
            
            if styles2 is not None:
                s_feats_mask = self.get_feats(styles2, all_layers)
            elif s_mask:
                masked_s = crop_nonzero_region(styles * s_mask)
                s_feats_mask = self.get_feats(masked_s, all_layers)
            else:
                s_feats_mask = self.get_feats(styles, all_layers)

        if 'lum_nnfm_loss' in loss_names:
            lum_outputs = lum_transform(outputs)
            lum_styles = lum_transform(styles)
            lum_x_feats = self.get_feats(lum_outputs, all_layers)
            lum_s_feats = self.get_feats(lum_styles, all_layers)

        ix_map = {}
        for a, b in enumerate(all_layers):
            ix_map[b] = a


        loss_dict = dict([(x, 0.) for x in loss_names])
        for block in blocks:
            layers = block_indexes[block]
            x_feats = jt.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
            s_feats = jt.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)
            # print('x_feat', x_feats.shape, 's_feat', s_feats.shape) # x_feat jt.Size([1, 768, 299, 400]) s_feat jt.Size([1, 768, 291, 400])

            if 'lum_nnfm_loss' in loss_names:
                x_feats = jt.cat([lum_x_feats[ix_map[ix]] for ix in layers], 1)
                s_feats = jt.cat([lum_s_feats[ix_map[ix]] for ix in layers], 1)
                target_feats = nn_feat_replace(x_feats, s_feats)
                loss_dict["lum_nnfm_loss"] += cos_loss(x_feats, target_feats)

            if "nnfm_loss" in loss_names:
                target_feats = nn_feat_replace(x_feats, s_feats)
                loss_dict["nnfm_loss"] += cos_loss(x_feats, target_feats)

            if "gram_loss" in loss_names:
                # loss_dict["gram_loss"] += jt.mean((gram_matrix(x_feats) - gram_matrix(s_feats)) ** 2)
                loss_dict["gram_loss"] += nn.mse_loss(gram_matrix(x_feats), gram_matrix(s_feats))

            if "content_loss" in loss_names:
                content_feats = jt.cat([content_feats_all[ix_map[ix]] for ix in layers], 1)
                # loss_dict["content_loss"] += jt.mean((content_feats - x_feats) ** 2)
                loss_dict["content_loss"] += nn.mse_loss(x_feats, content_feats)
                # loss_dict["content_loss"] += content_loss(x_feats, content_feats)

            if "spatial_loss" in loss_names:
                x_mask_feats = jt.cat([x_feats_mask[ix_map[ix]] for ix in layers], 1)
                s_mask_feats = jt.cat([s_feats_mask[ix_map[ix]] for ix in layers], 1)
                # print('x_feat mask', x_mask_feats.shape, 's_feat mask', s_mask_feats.shape) # x_feat mask jt.Size([1, 768, 299, 400]) s_feat mask jt.Size([1, 768, 291, 400])
                # loss_dict['spatial_loss'] += F.mse_loss(gram_matrix(x_mask_feats), gram_matrix(s_mask_feats))
                loss_dict['spatial_loss'] += cos_loss(x_mask_feats, nn_feat_replace(x_mask_feats, s_mask_feats))

        if "scale_loss" in loss_names:
            for layerindex in all_layers:
                x_feats = x_feats_all[ix_map[layerindex]]
                s_feats = s_feats_all[ix_map[layerindex]]
                if layer_coef[ix_map[layerindex]] != 0:
                    target_feats = nn_feat_replace(x_feats, s_feats)
                    loss_dict["scale_loss"] += cos_loss(x_feats, target_feats)*layer_coef[ix_map[layerindex]]


        return loss_dict


""" VGG-16 Structure
Input image is [-1, 3, 224, 224]
-------------------------------------------------------------------------------
        Layer (type)               Output Shape         Param #     Layer index
===============================================================================
            Conv2d-1         [-1, 64, 224, 224]           1,792     
              ReLU-2         [-1, 64, 224, 224]               0               1
            Conv2d-3         [-1, 64, 224, 224]          36,928     
              ReLU-4         [-1, 64, 224, 224]               0               3
         MaxPool2d-5         [-1, 64, 112, 112]               0     
            Conv2d-6        [-1, 128, 112, 112]          73,856     
              ReLU-7        [-1, 128, 112, 112]               0               6
            Conv2d-8        [-1, 128, 112, 112]         147,584     
              ReLU-9        [-1, 128, 112, 112]               0               8
        MaxPool2d-10          [-1, 128, 56, 56]               0     
           Conv2d-11          [-1, 256, 56, 56]         295,168     
             ReLU-12          [-1, 256, 56, 56]               0              11
           Conv2d-13          [-1, 256, 56, 56]         590,080     
             ReLU-14          [-1, 256, 56, 56]               0              13
           Conv2d-15          [-1, 256, 56, 56]         590,080     
             ReLU-16          [-1, 256, 56, 56]               0              15
        MaxPool2d-17          [-1, 256, 28, 28]               0     
           Conv2d-18          [-1, 512, 28, 28]       1,180,160     
             ReLU-19          [-1, 512, 28, 28]               0              18
           Conv2d-20          [-1, 512, 28, 28]       2,359,808     
             ReLU-21          [-1, 512, 28, 28]               0              20
           Conv2d-22          [-1, 512, 28, 28]       2,359,808     
             ReLU-23          [-1, 512, 28, 28]               0              22
        MaxPool2d-24          [-1, 512, 14, 14]               0     
           Conv2d-25          [-1, 512, 14, 14]       2,359,808     
             ReLU-26          [-1, 512, 14, 14]               0              25
           Conv2d-27          [-1, 512, 14, 14]       2,359,808     
             ReLU-28          [-1, 512, 14, 14]               0              27
           Conv2d-29          [-1, 512, 14, 14]       2,359,808    
             ReLU-30          [-1, 512, 14, 14]               0              29
        MaxPool2d-31            [-1, 512, 7, 7]               0    
===============================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 218.39
Params size (MB): 56.13
Estimated Total Size (MB): 275.10
----------------------------------------------------------------
"""


if __name__ == '__main__':
    import imageio
    import cv2
    import numpy as np
    style_img = imageio.imread("/mnt/155_16T/zhangbotao/jgaussian/StylizedGSjittor/data/starry_night.jpg", pilmode="RGB").astype(np.float32) / 255.0 # pilmode="RGB"
    style_h, style_w = style_img.shape[:2]
    content_long_side = max([256,256])
    if style_h > style_w:
        style_img = cv2.resize(
            style_img,
            (int(content_long_side / style_h * style_w), content_long_side),
            interpolation=cv2.INTER_AREA,
        )
    else:
        style_img = cv2.resize(
            style_img,
            (content_long_side, int(content_long_side / style_w * style_h)),
            interpolation=cv2.INTER_AREA,
        )
    style_img = cv2.resize(
        style_img,
        (style_img.shape[1] // 2, style_img.shape[0] // 2),
        interpolation=cv2.INTER_AREA,
    )

    fake_style = jt.array(style_img).permute(2,0,1).unsqueeze(0)
    device = "cuda"
    nnfm_loss_fn = NNFMLoss(device)
    fake_output = jt.ones(1, 3, 256, 256)*0.3
    # fake_style = jt.ones(1, 3, 256, 256)*0.5
    fake_content = jt.ones(1, 3, 256, 256)*0.7

    loss = nnfm_loss_fn(outputs=fake_output, styles=fake_style, contents=fake_content, loss_names=["nnfm_loss", "content_loss", "gram_loss"])
    ic(loss)
    ic(jt.grad(loss["nnfm_loss"],fake_output))
    # for name, param in nnfm_loss_fn.named_parameters():
    #     print(name)
    fake_image_set = jt.ones(10, 256, 256, 3)*0.3
    fake_style = jt.ones(256, 256, 3)*0.7
    fake_image_set_new, color_tf = match_colors_for_image_set(fake_image_set, fake_style)
    fake_image_set_new, color_tf = color_histgram_match(fake_image_set, fake_style)
    ic(color_tf)
    ic(fake_image_set_new.shape, color_tf.shape)
