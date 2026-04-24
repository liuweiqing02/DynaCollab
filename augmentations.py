import torch
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import transform as sk_tf
from collections import namedtuple
import numpy as np
import numbers


def interval(obj, lower=None):
    """ Listify an object.

    Parameters
    ----------
    obj: 2-uplet or number
        the object used to build the interval.
    lower: number, default None
        the lower bound of the interval. If not specified, a symetric
        interval is generated.

    Returns
    -------
    interval: 2-uplet
        an interval.
    """
    if isinstance(obj, numbers.Number):
        if obj < 0:
            raise ValueError("Specified interval value must be positive.")
        if lower is None:
            lower = -obj
        return (lower, obj)
    if len(obj) != 2:
        raise ValueError("Interval must be specified with 2 values.")
    min_val, max_val = obj
    if min_val > max_val:
        raise ValueError("Wrong interval boudaries.")
    return tuple(obj)


class Transformer(object):
    """ Class that can be used to register a sequence of transformations.
    """
    Transform = namedtuple("Transform", ["transform", "probability"])

    def __init__(self):
        """ Initialize the class.
        """
        self.transforms = []

    def register(self, transform, probability=1):
        """ Register a new transformation.
        Parameters
        ----------
        transform: callable
            the transformation object.
        probability: float, default 1
            the transform is applied with the specified probability.
        """
        trf = self.Transform(transform=transform, probability=probability, )
        self.transforms.append(trf)

    def __call__(self, arr):
        """ Apply the registered transformations.
        """
        transformed = arr.copy()
        for trf in self.transforms:
            if np.random.rand() < trf.probability:
                transformed = trf.transform(transformed)
        return transformed

    def __str__(self):
        if len(self.transforms) == 0:
            return '(Empty Transformer)'
        s = 'Composition of:'
        for trf in self.transforms:
            s += '\n\t- '+trf.__str__()
        return s


class Normalize(object):
    def __init__(self, mean=0.0, std=1.0, eps=1e-8):
        self.mean=mean
        self.std=std
        self.eps=eps

    def __call__(self, arr, is_label=False):
        return self.std * (arr - np.mean(arr))/(np.std(arr) + self.eps) + self.mean


class Crop(object):
    """Crop the given n-dimensional array either at a random location or centered"""

    def __init__(self, shape, type="center", resize=False, keep_dim=False, is_label=False):
        """
        :param shape: tuple or list of int
            The shape of the patch to crop
        :param type: 'center' or 'random'
            Whether the crop will be centered or at a random location
        :param resize: bool, default False
            If True, resize the cropped patch to the inital dim.
        :param keep_dim: bool, default False
            if True and resize==False, put a constant value around the patch cropped.
        :param is_label: bool, default False
            Whether this crop is applied to label data
        """
        assert type in ["center", "random"]
        self.shape = shape
        self.copping_type = type
        self.resize = resize
        self.keep_dim = keep_dim
        self.is_label = is_label  # 新增：标识是否用于标签

    def __call__(self, arr, is_label=False):
        assert isinstance(arr, np.ndarray)
        if not isinstance(self.shape, int):
            shape_len = len(self.shape)
            if shape_len not in {len(arr.shape), len(arr.shape) - 1}:
                raise AssertionError("Shape of array {} does not match {}".format(arr.shape, self.shape))

        # 记录原始数据类型
        original_dtype = arr.dtype

        img_shape = np.array(arr.shape)
        if type(self.shape) == int:
            size = [self.shape for _ in range(len(img_shape))]
        else:
            size = np.copy(self.shape)
            if len(size) == len(img_shape) - 1:
                # Treat provided shape as spatial-only and keep channel axis unchanged.
                size = np.concatenate(([img_shape[0]], size))

        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.copping_type == "center":
                delta_before = (img_shape[ndim] - size[ndim]) / 2.0
            elif self.copping_type == "random":
                delta_before = np.random.randint(0, img_shape[ndim] - size[ndim] + 1)
            indexes.append(slice(int(delta_before), int(delta_before + size[ndim])))

        cropped = arr[tuple(indexes)]

        if self.resize:
            # 对于标签数据，使用最近邻插值保持整数
            if is_label or self.is_label:
                # 使用最近邻插值
                resized = sk_tf.resize(
                    cropped,
                    img_shape,
                    order=0,  # 最近邻插值
                    preserve_range=True,
                    anti_aliasing=False
                )
                # 确保数据类型与原始一致
                return resized.astype(original_dtype)
            else:
                # 对于图像数据，使用默认插值
                return sk_tf.resize(cropped, img_shape, preserve_range=True)

        if self.keep_dim:
            mask = np.zeros(img_shape, dtype=bool)
            mask[tuple(indexes)] = True
            arr_copy = arr.copy()
            arr_copy[~mask] = 0
            return arr_copy

        return cropped


class Cutout(object):
    """Apply a cutout on the images
    cf. Improved Regularization of Convolutional Neural Networks with Cutout, arXiv, 2017
    We assume that the square to be cut is inside the image.
    """
    def __init__(self, patch_size=None, value=0, random_size=False, inplace=False, localization=None):
        self.patch_size = patch_size
        self.value = value
        self.random_size = random_size
        self.inplace = inplace
        self.localization = localization

    def __call__(self, arr, is_label=False):

        img_shape = np.array(arr.shape)
        if type(self.patch_size) == int:
            size = [self.patch_size for _ in range(len(img_shape))]
        else:
            size = np.copy(self.patch_size)
        if len(size) == len(img_shape) - 1:
            # Spatial-only patch size: keep channel dimension intact.
            size = np.concatenate(([img_shape[0]], size))
        assert len(size) == len(img_shape), "Incorrect patch dimension."
        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.random_size:
                size[ndim] = np.random.randint(0, size[ndim])
            if self.localization is not None:
                delta_before = max(self.localization[ndim] - size[ndim]//2, 0)
            else:
                delta_before = np.random.randint(0, img_shape[ndim] - size[ndim] + 1)
            indexes.append(slice(int(delta_before), int(delta_before + size[ndim])))
        if self.inplace:
            arr[tuple(indexes)] = self.value
            return arr
        else:
            arr_cut = np.copy(arr)
            arr_cut[tuple(indexes)] = self.value
            return arr_cut

class Flip(object):
    """ Apply a random mirror flip."""
    def __init__(self, axis=None):
        '''
        :param axis: int, default None
            apply flip on the specified axis. If not specified, randomize the
            flip axis.
        '''
        self.axis = axis

    def __call__(self, arr, is_label=False):
        if self.axis is not None:
            axis = self.axis
        elif arr.ndim >= 4:
            # Channel-first arrays: only flip spatial dimensions.
            axis = np.random.randint(low=1, high=arr.ndim, size=1)[0]
        else:
            axis = np.random.randint(low=0, high=arr.ndim, size=1)[0]
        return np.flip(arr, axis=axis)


class Blur(object):
    def __init__(self, snr=None, sigma=None):
        """ Add random blur using a Gaussian filter.
            Parameters
            ----------
            snr: float, default None
                the desired signal-to noise ratio used to infer the standard deviation
                for the noise distribution.
            sigma: float or 2-uplet
                the standard deviation for Gaussian kernel.
        """
        if snr is None and sigma is None:
            raise ValueError("You must define either the desired signal-to noise "
                             "ratio or the standard deviation for the noise "
                             "distribution.")
        self.snr = snr
        self.sigma = sigma

    def __call__(self, arr, is_label=False):
        sigma = self.sigma
        if self.snr is not None:
            s0 = np.std(arr)
            sigma = s0 / self.snr
        sigma = interval(sigma, lower=0)
        sigma_random = np.random.uniform(low=sigma[0], high=sigma[1], size=1)[0]
        return gaussian_filter(arr, sigma_random)


class Noise(object):
    def __init__(self, snr=None, sigma=None, noise_type="gaussian"):
        """ Add random Gaussian or Rician noise.

           The noise level can be specified directly by setting the standard
           deviation or the desired signal-to-noise ratio for the Gaussian
           distribution. In the case of Rician noise sigma is the standard deviation
           of the two Gaussian distributions forming the real and imaginary
           components of the Rician noise distribution.

           In anatomical scans, CNR values for GW/WM ranged from 5 to 20 (1.5T and
           3T) for SNR around 40-100 (http://www.pallier.org/pdfs/snr-in-mri.pdf).

           Parameters
           ----------
           snr: float, default None
               the desired signal-to noise ratio used to infer the standard deviation
               for the noise distribution.
           sigma: float or 2-uplet, default None
               the standard deviation for the noise distribution.
           noise_type: str, default 'gaussian'
               the distribution of added noise - can be either 'gaussian' for
               Gaussian distributed noise, or 'rician' for Rice-distributed noise.
        """

        if snr is None and sigma is None:
            raise ValueError("You must define either the desired signal-to noise "
                             "ratio or the standard deviation for the noise "
                             "distribution.")
        assert noise_type in {"gaussian", "rician"}, "Noise muse be either Rician or Gaussian"
        self.snr = snr
        self.sigma = sigma
        self.noise_type = noise_type


    def __call__(self, arr, is_label=False):
        sigma = self.sigma
        if self.snr is not None:
            s0 = np.std(arr)
            sigma = s0 / self.snr
        sigma = interval(sigma, lower=0)
        sigma_random = np.random.uniform(low=sigma[0], high=sigma[1], size=1)[0]
        noise = np.random.normal(0, sigma_random, [2] + list(arr.shape))
        if self.noise_type == "gaussian":
            transformed = arr + noise[0]
        elif self.noise_type == "rician":
            transformed = np.square(arr + noise[0])
            transformed += np.square(noise[1])
            transformed = np.sqrt(transformed)
        return transformed


import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.ndimage
import numbers


class Rotate3D(object):
    """3D rotation transformation with synchronized parameters support"""

    def __init__(self, angles=(30, 30, 30), order=0, mode='constant', cval=0.0):
        self.angle_ranges = [interval(a) for a in angles]
        self.order = order
        self.mode = mode
        self.cval = cval

        # Will be set during call
        self.current_angles = None
        self.rotation_matrix = None
        self.offset = None

    def __call__(self, arr, shared_params=None, is_label=False):
        """
        arr: 3D numpy array (H, W, D)
        shared_params: dictionary for parameter synchronization
        is_label: flag for label channel processing
        """
        # Get parameters from shared params or generate new
        if shared_params and 'rotate' in shared_params:
            params = shared_params['rotate']
            angles = params['angles']
            R_matrix = params['matrix']
            offset = params['offset']
        else:
            # Generate new parameters
            angles = [np.random.uniform(low, high) for low, high in self.angle_ranges]

            # Create rotation matrix
            rot = R.from_euler('zyx', angles, degrees=True)
            R_matrix = rot.as_matrix()

            # Calculate offset to keep center position
            spatial_shape = arr.shape
            center = np.array([(d - 1) / 2 for d in spatial_shape])
            offset = center - R_matrix.dot(center)

            # Save to shared params if provided
            if shared_params is not None:
                shared_params['rotate'] = {
                    'angles': angles,
                    'matrix': R_matrix,
                    'offset': offset
                }

        # Use nearest neighbor interpolation for labels
        current_order =0 if is_label else 1

        # Apply rotation
        rotated = scipy.ndimage.affine_transform(
            arr,
            matrix=R_matrix,
            offset=offset,
            order=current_order,
            mode=self.mode,
            cval=self.cval
        )
        return rotated

class ElasticTransform3D:
    """3D弹性形变增强（与Rotate3D类似，支持参数同步）"""

    def __init__(self, alpha_range=(30, 50), sigma=10, mode='constant', cval=0.0):
        """
        :param alpha_range: 形变强度范围（值越大形变越强）
        :param sigma: 高斯滤波的sigma（控制形变的平滑程度）
        :param mode: 边界处理模式（同scipy）
        :param cval: 填充值（当mode='constant'时有效）
        """
        self.alpha_range = interval(alpha_range, lower=0)
        self.sigma = sigma
        self.mode = mode
        self.cval = cval

        # 共享参数存储
        self.current_alpha = None
        self.displacement = None

    def __call__(self, arr, shared_params=None, is_label=False):
        """
        arr: 单通道三维数组 [D, H, W]
        is_label: 是否为标签数据（决定插值方式）
        """
        if shared_params and 'elastic' in shared_params:
            displacement = shared_params['elastic']['displacement']
        else:
            # 生成新的位移场（仅在第一次调用时生成）
            np.random.seed()  # 避免全局种子干扰
            alpha = np.random.uniform(*self.alpha_range)
            shape = arr.shape  # [D, H, W]

            dx = gaussian_filter(
                (np.random.rand(*shape) * 2 - 1),
                self.sigma, mode='constant', cval=0) * alpha
            dy = gaussian_filter(
                (np.random.rand(*shape) * 2 - 1),
                self.sigma, mode='constant', cval=0) * alpha
            dz = gaussian_filter(
                (np.random.rand(*shape) * 2 - 1),
                self.sigma, mode='constant', cval=0) * alpha
            displacement = np.stack([dz, dy, dx], axis=0)  # [3, D, H, W]

            if shared_params is not None:
                shared_params['elastic'] = {'displacement': displacement}

        # 应用位移场
        return self._apply_displacement(arr, displacement, order=0 if is_label else 1)

    def _apply_displacement(self, data, displacement, order):
        """应用位移场到单通道数据"""
        D, H, W = data.shape
        z, y, x = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
        indices = [
            (z + displacement[0]).reshape(-1, 1),
            (y + displacement[1]).reshape(-1, 1),
            (x + displacement[2]).reshape(-1, 1)
        ]
        return map_coordinates(
            data,
            indices,
            order=order,
            mode=self.mode,
            cval=self.cval
        ).reshape(D, H, W)

# class SafeTransformer(object):
#     """改进后的增强处理器，支持同步和独立增强"""
#     Transform = namedtuple("Transform", ["transform", "probability", "apply_to"])
#
#     def __init__(self):
#         self.transforms = []
#
#     def register(self, transform, probability=1, apply_to='image'):
#         """注册增强方法，指定作用对象"""
#         trf = self.Transform(transform=transform, probability=probability, apply_to=apply_to)
#         self.transforms.append(trf)
#
#     def __call__(self, data):
#         # 解包数据并判断是否有标签
#         if len(data) == 1:
#             image = data[0]
#             label = None
#         else:
#             image = data[0]
#             label = data[1]
#
#         for trf in self.transforms:
#             if np.random.rand() < trf.probability:
#                 if trf.apply_to == 'both':
#                     # 同步变换处理（图像和标签）
#                     if label is not None:
#                         seed = np.random.randint(0, 2 ** 32)
#                         # 处理图像
#                         np.random.seed(seed)
#                         torch.manual_seed(seed)
#                         transformed_image = trf.transform(image,is_label=False)
#                         # 处理标签（使用相同种子）
#                         np.random.seed(seed)
#                         torch.manual_seed(seed)
#                         transformed_label = trf.transform(label,is_label=True)
#                         image, label = transformed_image, transformed_label
#                     else:
#                         # 只有图像时单独处理
#                         seed = np.random.randint(0, 2 ** 32)
#                         np.random.seed(seed)
#                         torch.manual_seed(seed)
#                         image = trf.transform(image, is_label=False)
#                 elif trf.apply_to == 'image':
#                     image = trf.transform(image, is_label=False)
#                 elif trf.apply_to == 'label':
#                     if label is not None:
#                         label = trf.transform(label, is_label=True)
#                 # unique_vals = np.unique(label)
#                 # print(unique_vals)
#
#         # 保持维度一致性
#         image = np.expand_dims(image, axis=0)
#         if label is not None:
#             label = np.expand_dims(label, axis=0)
#             return np.concatenate([image, label], axis=0)
#         else:
#             return image

class SafeTransformer(object):
    """改进后的增强处理器，支持同步和独立增强"""
    Transform = namedtuple("Transform", ["transform", "probability", "apply_to"])

    def __init__(self):
        self.transforms = []

    def register(self, transform, probability=1, apply_to='image'):
        """注册增强方法，指定作用对象"""
        trf = self.Transform(transform=transform, probability=probability, apply_to=apply_to)
        self.transforms.append(trf)

    def __call__(self, data):
        # 解包数据并判断是否有标签
        if len(data) == 1:
            image = data
            label = None
        else:
            num_channels = data.shape[0]
            image_channels = num_channels - 1
            image = data[:image_channels, ...]  # 图像部分 [image_channels, D, H, W]
            label = data[image_channels:, ...]  # 标签部分 [1, D, H, W]

        for trf in self.transforms:
            if np.random.rand() < trf.probability:
                if trf.apply_to == 'both':
                    # 同步变换处理（图像和标签）
                    if label is not None:
                        seed = np.random.randint(0, 2 ** 32)
                        # 处理图像
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        transformed_image = trf.transform(image,is_label=False)
                        # 处理标签（使用相同种子）
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        transformed_label = trf.transform(label,is_label=True)
                        image, label = transformed_image, transformed_label
                    else:
                        # 只有图像时单独处理
                        seed = np.random.randint(0, 2 ** 32)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        image = trf.transform(image, is_label=False)
                elif trf.apply_to == 'image':
                    image = trf.transform(image, is_label=False)
                elif trf.apply_to == 'label':
                    if label is not None:
                        label = trf.transform(label, is_label=True)
                # unique_vals = np.unique(label)
                # print(unique_vals)

        # 保持维度一致性
        if label is not None:
            return np.concatenate([image, label], axis=0)
        else:
            return image

if __name__ == '__main__':
    transformer = Transformer()
    # 每个轴旋转角度范围：x(-15,15)，y(-20,20)，z(-10,10)
    transformer.register(Rotate(angles=(15, 20, 10)), probability=0.5)

    # 应用变换
    input_data = np.random.rand(3, 32, 32, 32)  # 假设通道在前
    output_data = transformer(input_data)
