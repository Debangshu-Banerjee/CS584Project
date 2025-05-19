import torch
from src.common import Dataset, RavenMode
from src.specLoader import get_specification, get_std
from src.netLoader import get_net
from src.adaptiveRavenBackend import AdaptiveRavenBackend
from src.adaptiveRavenResult import AdaptiveRavenResultList
from raven.src.config import mnist_data_transform
import raven.src.config as config
from auto_LiRPA.operators import BoundLinear, BoundConv
import numpy as np
from raven.src.network_conversion_helper import get_pytorch_net
from auto_LiRPA import BoundedModule, PerturbationLpNorm, BoundedTensor
from src.m import MultiNetMILPTransformer


class Property:
    def __init__(self, inputs, labels, eps, constraint_matrices, lbs, ubs) -> None:
        self.inputs = inputs
        self.labels = labels
        self.eps = eps
        self.constraint_matrices = constraint_matrices
        self.lbs = lbs
        self.ubs = ubs

def shift_props_to_device(prop, device):
    prop.inputs = prop.inputs.to(device)
    prop.labels = prop.labels.to(device)
    prop.constraint_matrices = prop.constraint_matrices.to(device)
    prop.lbs = prop.lbs.to(device)
    prop.ubs = prop.ubs.to(device)

def shift_to_device(device, models, prop):
    shift_props_to_device(prop=prop, device=device)
    for i, model in enumerate(models):
        models[i] = model.to(device) 

def populate_names(model):
    final_names = []
    input_names = []
    layer_names = []
    i = 0
    last_name = None
    for node_name, node in model._modules.items():
        if i == 0:
            input_names.append(node_name)
        i += 1
        if type(node) in [BoundLinear, BoundConv]:
            layer_names.append(node_name)
            last_name = node_name
    assert last_name is not None
    final_names.append(node_name)
    return input_names, layer_names, final_names

def get_linear_coefs_biases(model, prop, device, input_name, final_name):
    ptb = PerturbationLpNorm(norm = np.inf, x_L=prop.lbs, x_U=prop.ubs)
    bounded_images = BoundedTensor(prop.inputs, ptb)
    coef_dict = {final_name: [input_name]}
    result = model.compute_bounds(x=(bounded_images,), method='CROWN-Optimized', C=prop.constraint_matrices,
                                    bound_upper=False, return_A=True, needed_A_dict=coef_dict, 
                                    multiple_execution=False, execution_count=None, ptb=ptb, 
                                    unperturbed_images = prop.inputs)
    lower_bnd, upper, A_dict = result
    lA = A_dict[final_name][input_name]['lA']
    lbias = A_dict[final_name][input_name]['lbias']
    lA = torch.reshape(lA,(1, 9,-1))
    return lA, lbias, lower_bnd


def multiNetworkRelationalProp(nets, prop, device='cuda:2'):
    final_names = []
    input_names = []
    models = []
    for net in nets:
        torch_net = get_pytorch_net(model=net, remove_last_layer=False, all_linear=False)
        models.append(BoundedModule(torch_net, (prop.inputs), bound_opts={}))
        

    for net in models:
        ins, _, fns = populate_names(net)
        final_names.append(fns[0])
        input_names.append(ins[0])
    shift_to_device(device=device, models=models, prop=prop)
    lAs, lbiases = [], []
    for i, model in enumerate(models):
        lA, lbias, lower_bnd = get_linear_coefs_biases(model, prop, device, input_names[i], final_names[i])
        lAs.append(lA)
        lbiases.append(lbias)
        print(f"la shape {lA.shape}")
        print(f"lbias shape {lbias.shape}")
    



if __name__ == "__main__":
    dataset = Dataset.CIFAR10
    net_names = [config.CIFAR_01, config.CIFAR_02, config.CIFAR_03]
    nets = get_net(net_names = net_names, dataset = dataset)
    prop_count = 1
    count_per_prop = 1
    total_input_count = 1
    eps = 2/255
    images, labels, constraint_matrices, lbs, ubs = get_specification(dataset=dataset,
                                                            raven_mode=RavenMode.UAP, 
                                                            count=total_input_count, nets=nets, eps=eps,
                                                            dataloading_seed=1232,
                                                            net_names=net_names,
                                                            only_cutoff=False)
    
    for i in range(prop_count):
        start = i * count_per_prop
        end = start + count_per_prop
        prop_images, prop_labels, prop_constraint_matrices = images[start:end], labels[start:end], constraint_matrices[start:end]
        prop_lbs, prop_ubs = lbs[start:end], ubs[start:end]
        prop = Property(inputs=prop_images, labels=prop_labels, 
                        eps=eps / get_std(dataset=dataset, transform=False),
                        constraint_matrices=prop_constraint_matrices, lbs=prop_lbs, ubs=prop_ubs)
        multiNetworkRelationalProp(nets=nets, prop=prop)
        # verifier = AdaptiveRavenBackend(prop=prop, nets=nets, args=raven_args)  
    pass