from raven.src.specs.input_spec import InputSpecType
from enum import IntEnum
from src.common import Dataset

NET_HOME = "/share/cs584_25/project_networks/"
DEVICE = 'cpu'
GPU_DEVICE_LIST = []


class Args:
    def __init__(self, net, domain, count=None, eps=0.01, dataset='mnist', spec_type=InputSpecType.LINF, split=None,
                 pt_method=None, timeout=None, parallel=False, initial_split=False, attack=None):
        self.net = NET_HOME + net
        self.domain = domain
        self.count = count
        self.eps = eps
        self.dataset = dataset
        self.spec_type = spec_type
        self.split = split
        self.pt_method = pt_method
        self.timeout = timeout
        self.parallel = parallel
        self.initial_split = initial_split
        self.attack = attack
        self.ignore_properties = []


class PruningArgs:
    def __init__(self, desried_perturbation=None, layers_to_prune=None, swap_layers= False, node_wise_bounds=False,
                        unstructured_pruning=True, structured_pruning=False, maximum_iteration=10, accuracy_drop=None):
        self.desired_perturbation= desried_perturbation
        self.layers_to_prune = layers_to_prune
        self.swap_layers = swap_layers
        self.node_wise_bounds= node_wise_bounds
        self.unstructured_pruning = unstructured_pruning
        self.structured_pruning = structured_pruning
        self.maximum_iteration = maximum_iteration
        self.accuracy_drop = accuracy_drop

log_file = "log.txt"
log_enabled = False


def write_log(log):
    """Appends string @param: str to log file"""
    if log_enabled:
        f = open(log_file, "a")
        f.write(log + '\n')
        f.close()


tool_name = "IVAN"
baseline = "Baseline"

MNIST_IBP_01 = "mnist_cnn_2layer_w1_ibp.onnx"
MNIST_IBP_02 = "mnist_cnn_2layer_w2_ibp.onnx"
MNIST_IBP_03 = "mnist_cnn_3layer_k3_w2_ibp.onnx"
MNIST_IBP_04 = "mnist_cnn_3layer_k5_w1_ibp.onnx"
MNIST_IBP_05 = "mnist_cnn_3layer_k5_w2.onnx"

MNIST_ENSEMBLE_01 = "ensemble_models/mnist_cnn_2layer_w1_ensemble.onnx"
MNIST_ENSEMBLE_02 = "ensemble_models/mnist_cnn_2layer_w2_ensemble.onnx"
MNIST_ENSEMBLE_03 = "ensemble_models/mnist_cnn_3layer_k3_w2_ensemble.onnx"
MNIST_ENSEMBLE_04 = "ensemble_models/mnist_cnn_3layer_k5_w1_ensemble.onnx"
MNIST_ENSEMBLE_05 = "ensemble_models/mnist_cnn_3layer_k5_w2_ensemble.onnx"
CIFAR_01 = "new_sabr/cifar10_cnn_2layer_w1_sabr.onnx"
CIFAR_02 = "new_sabr/cifar10_cnn_2layer_w2_sabr.onnx"
CIFAR_03 = "new_sabr/cifar10_cnn_3layer_k3_w2_sabr.onnx"
CIFAR_04 = "new_sabr/cifar10_cnn_3layer_k5_w1_sabr.onnx"
CIFAR_05 = "new_sabr/cifar10_cnn_3layer_k5_w2_sabr.onnx"

# Networks
MNIST_FFN_01 = "mnist_0.1.onnx"  # 9508 neuron
MNIST_FFN_03 = "mnist_0.3.onnx"
MNIST_CONV_PGD_03 = "convMedGRELU__PGDK_w_0.3.onnx"
MNIST_FFN_L2 = "mnist-net_256x2.onnx"  # 512 neurons
MNIST_FFN_L4 = "mnist-net_256x4.onnx"   # 1024 neurons
MNIST_FFN_torch1 = "cpt/fc1.pt"
MNIST_STANDARD_MODIFIED = "mnist_standard_modified.pt"
MNIST_LINEAR_5_100 = "mnist_relu_5_100.onnx"
MNIST_BINARY = "mnist_binary.onnx"
MNIST_BINARY_PGD_RELU = "mnist_binary_relu_pgd.pt"
MNIST_BINARY_PGD_SIGMOID = "mnist_binary_sigmoid_pgd.pt"
MNIST_BINARY_PGD_TANH = "mnist_binary_tanh_pgd.pt"

MNIST_LINEAR_50 = "mnist_relu_3_50.onnx"
MNIST_LINEAR_100 = "mnist_relu_3_100.onnx"
MNIST_LINEAR_6_100 = "mnist_relu_6_100.onnx"
MNIST_LINEAR_9_200 = "mnist_relu_9_200.onnx"

MNIST_CONV_PGD = "mnistconvSmallRELU__PGDK.onnx"
MNIST_CONV_SMALL = 'mnist_convSmallRELU__Point.onnx'
MNIST_FFN_PGD = "mnistconvSmallRELU__PGDK.onnx"
MNIST_CONV_SMALL_DIFFAI = "mnistconvSmallRELUDiffAI.onnx"
MNIST_FFN_DIFFAI = "mnistconvSmallRELUDiffAI.onnx"
MNIST_CONV_MED = 'mnistconvMedGRELU__Point.onnx'
MNIST_CONV_BIG = 'mnist_convBigRELU__DiffAI.onnx'
MNIST_CONV_SIGMOID = 'convMedGSIGMOID__PGDK_w_0.3.onnx'
MNIST_CONV_TANH = "convMedGTANH__PGDK_w_0.3.onnx"
MNIST_FFN_SIGMOID = "ffnnSIGMOID__PGDK_w_0.3_6_500.onnx"

# MNIST crown-ibp network
MNIST_CROWN_IBP = "mnist_cnn_2layer_width_1_best.pth"
MNIST_CROWN_IBP_MED = "mnist_cnn_3layer_fixed_kernel_3_width_1_best.pth"
MNIST_CROWN_IBP_MODIFIED = "mnist_cnn_2layer_width_1_best_modified.pth"
# MNIST CITRUS
MNIST_CONV_SMALL_CITRUS_1 = "citrus_convsmall_1.onnx"
# MNIST SABR
MNIST_CONV_SMALL_SABR_1 = "sabr_convsmall_1.onnx"


# CIFAR10 CITRUS
CIFAR_CITRUS_2 = "citrus_convsmall_2.onnx"

# CIFAR10 SABR
CIFAR_SABR_2 = "sabr_convsmall_2.onnx"

# CIFAR Common Network
CIFAR_STANDARD_4 = "cifar_relu_4_100.onnx"
CIFAR_STANDARD_6 = "ffnnRELU__Point_6_500.onnx"
CIFAR_STANDARD_CONV = "convSmallRELU__Point.onnx"

# CIFAR crown-ibp Network
CIFAR_CROWN_IBP = "cifar_cnn_2layer_width_2_best.pth"
CIFAR_CROWN_IBP_MEDIUM = "cifar_cnn_3layer_fixed_kernel_3_width_1_best.pth"

# CIFAR Networks
CIFAR_CONV_2_255 = "cifar10_2_255_colt.onnx"  # 49402 neurons
CIFAR_CONV_8_255 = "cifar10_8_255.onnx"  # 16634 neurons
CIFAR_CONV_COLT = "cifar10_8_255_colt.onnx"
CIFAR_CONV_DIFFAI = "cifar10convSmallRELUDiffAI.onnx"

"cifar10_cnn_2layer_w1_sabr.onnx"
"cifar10_cnn_2layer_w2_sabr.onnx"
"cifar10_cnn_3layer_k3_w2_sabr.onnx"
"cifar10_cnn_3layer_k5_w1_sabr.onnx"
"cifar10_cnn_3layer_k5_w2_sabr.onnx"

# OVAL21 CIFAR
CIFAR_OVAL_BASE = "oval21/cifar_base_kw.onnx"   # 3172 neurons
CIFAR_OVAL_WIDE = "oval21/cifar_wide_kw.onnx"   # 6244 neurons
CIFAR_OVAL_DEEP = "oval21/cifar_deep_kw.onnx"   # 6756 neurons

CIFAR_CONV_SMALL = "convSmallRELU__Point.onnx"   # 3,604 neurons
CIFAR_CONV_SMALL_PGD = "convSmall_pgd_cifar.onnx"   # 3,604 neurons
CIFAR_CONV_SMALL_DIFFAI = "cifar10convSmallRELUDiffAI.onnx"   # 3,604 neurons
CIFAR_CONV_MED = "cifarconvmedRELU__Point.onnx"   # 3,604 neurons
CIFAR_CONV_BIG = "cifar_convBigRELU__DiffAI.onnx"


HOUSING_RM_CRIM = 'monotonic.onnx'
HOUSING_2LAYER = 'monotonic_2layer.onnx'
HOUSING_2LAYER_100 = 'monotonic_2layer_100.onnx'
HOUSING_2LAYER_200 = 'monotonic_2layer_200.onnx'

ADULT_TANH = 'cls_tanh.onnx'


linear_models = [MNIST_FFN_L2, MNIST_LINEAR_50, MNIST_LINEAR_100, MNIST_FFN_L4]

def is_linear_model(net_name):
    for name in linear_models:
        if name in net_name:
            return True
    return False

MNIST_NO_TRANSFORM_NETS = [MNIST_CROWN_IBP, MNIST_CROWN_IBP_MED, MNIST_CROWN_IBP_MODIFIED]


def mnist_data_transform(dataset, net_name):
    if dataset == Dataset.MNIST:
        for name in MNIST_NO_TRANSFORM_NETS:
            if name in net_name:
                return False
        return True
    elif dataset == Dataset.CIFAR10:
        return True
    else:
        raise ValueError(f'Unsupported dataset {dataset}')


class MONOTONE_PROP(IntEnum):
    CRIM = 0
    ZN = 1
    INDUS = 2
    NOX = 3
    RM = 4
    AGE = 5
    DIS = 6
    RAD = 7
    TAX = 8
    PTRATIO = 9
    B = 10
    LSTAT = 11


def ACASXU(i, j):
    net_name = "acasxu/nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
    return net_name
