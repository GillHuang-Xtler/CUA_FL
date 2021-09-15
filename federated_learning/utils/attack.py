import torch
import numpy as np

def reverse_nn_parameters(parameters, previous_weight, args):
    """
    generate reverse all layers parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """

    args.get_logger().info("Reverse all layers of gradients from attackers")
    new_parameters = []
    for params in parameters[:len(parameters)-args.get_num_attackers()]:
        new_parameters.append(params)

    for params in parameters[len(parameters)-args.get_num_attackers():]:
        for name in parameters[0].keys():
            # params[name] = (2*previous_weight[name].data - params[name].data)
            params[name] = - params[name].data
        new_parameters.append(params)

    return new_parameters

def reverse_last_parameters(parameters, previous_weight, args):
    """
    generate reverse last layers parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    args.get_logger().info("Reverse last layers of gradients from attackers")
    layers = list(parameters[0].keys())
    new_parameters = []
    for params in parameters[:len(parameters)-args.get_num_attackers()]:
        new_parameters.append(params)

    for params in parameters[len(parameters)-args.get_num_attackers():]:
        for name in parameters[0].keys():
            if name in layers[-(args.get_num_reverse_layers()):]:
                params[name] = (2*previous_weight[name].data - params[name].data)
            else:
                params[name] = params[name]
        new_parameters.append(params)

    return new_parameters

def lie_nn_parameters(parameters, args):
    """
    generate lie parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """

    args.get_logger().info("Averaging parameters on model lie attackers")
    new_parameters = []
    for params in parameters[:len(parameters)-args.get_num_attackers()]:
        new_parameters.append(params)

    # print("benign: "+ str(len(new_parameters)))

    z_value = args.get_lie_z_value()
    mean_params = {}
    std_params = {}
    for name in parameters[0].keys():
        mean_params[name] = sum([param[name].data for param in parameters])/len(parameters)
        _std_params = []
        for param in parameters:
            _std_params.append(param[name].data)
        val = torch.stack(_std_params, 0)
        std_params[name] = torch.std(val.float())

    mal_param = {}
    for name in parameters[0].keys():
        mal_param[name] = mean_params[name] + z_value[args.get_num_attackers()] * std_params[name]

    # [new_parameters.append(mal_param) for i in range(args.get_num_attackers())]
    new_parameters.append(mal_param)
    # print("all: "+ str(len(new_parameters)))

    return new_parameters

### Multi-krum and Bulyan
def fang_nn_parameters(parameters, args):
    """
    generate fang parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """

    args.get_logger().info("Averaging parameters on model fang attackers")
    new_parameters = []
    for params in parameters[:len(parameters)-args.get_num_attackers()]:
        new_parameters.append(params)

    z_value = args.get_lie_z_value()
    mean_params = {}
    std_params = {}
    deviation = {}
    for name in parameters[0].keys():
        mean_params[name] = sum([param[name].data for param in parameters])/len(parameters)
        deviation[name] = torch.sign(mean_params[name])

    lamda = compute_lambda_fang(parameters, mean_params, args.get_num_attackers())
    threshold = 1e-5
    mal_params = {}
    while lamda > threshold:
        for name in parameters[0].keys():
            mal_params[name] = (- lamda * deviation[name])
        new_parameters = torch.stack([mal_params] * args.get_num_attackers())
        new_parameters = torch.cat((new_parameters, parameters[:len(parameters)-args.get_num_attackers()]), 0)



    mal_param = {}
    for name in parameters[0].keys():
        mal_param[name] = mean_params[name] + z_value[args.get_num_attackers()] * std_params[name]

    [new_parameters.append(mal_param) for i in range(args.get_num_attackers())]

    return new_parameters

def ndss_nn_parameters(parameters,args):
    """
    generate ndss parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    import numpy as np
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.get_logger().info("Averaging parameters on model ndss attackers")

    model_re =torch.from_numpy(np.array(parameters[-1].values()).astype(float))
    all_updates = parameters[:-1]

    if args.get_dev_type() == 'unit_vec':
        deviation = model_re / torch.norm(model_re)
    elif args.get_dev_type() == 'sign':
        deviation = torch.sign(model_re)
    elif args.get_dev_type() == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([10.0]).to(device)

    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    distances = []
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    max_distance = torch.max(distances)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((all_updates - mal_update), dim=1) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)

    new_parameters = all_updates.extend(mal_update)

    new_params = {}
    for name in parameters[0].keys():
        new_params[name] = sum([param[name].data for param in all_updates])
        new_params[name] += (mal_update[name].data) * args.get_num_attackers()
        new_params[name] /= (len(all_updates) + args.get_num_attackers())

    return new_params

def free_nn_parameters(parameters, previous_weight, args):
    """
    generate reverse all layers parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """

    args.get_logger().info("Data Free Untargeted Attack")
    new_parameters = []
    for params in parameters[:len(parameters)-args.get_num_attackers()]:
        new_parameters.append(params)

    tmp = {}
    for name in previous_weight.keys():
        tmp[name] = (previous_weight[name].data)

    for i in range(args.get_num_attackers()):
        new_parameters.append(tmp)
    args.get_logger().info("the last 2 client do not have any data for training")

    return new_parameters