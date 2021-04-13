import torch
import numpy as np
from federated_learning.arguments import Arguments


def bulyan(all_updates, n_attackers):
    nusers = all_updates.shape[0]
    bulyan_cluster = []
    candidate_indices = []
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))

    while len(bulyan_cluster) < (nusers - 2 * n_attackers):
        torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]

        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]
        if not len(indices):
            break
        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        bulyan_cluster = remaining_updates[indices[0]][None, :] if not len(bulyan_cluster) else torch.cat((bulyan_cluster, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)

    # print('dim of bulyan cluster ', bulyan_cluster.shape)

    n, d = bulyan_cluster.shape
    param_med = torch.median(bulyan_cluster, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(bulyan_cluster - param_med), dim=0)
    sorted_params = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]

    return torch.mean(sorted_params[:n - 2 * n_attackers], dim=0), np.array(candidate_indices)

def multi_krum_nn_parameters(dict_parameters, args):
    """
    multi krum passed parameters.

    :param dict_parameters: nn model named parameters with client index
    :type parameters: list
    """
    from torch.nn import functional as F
    import numpy as np

    args.get_logger().info("Averaging parameters on multi krum")
    multi_krum = 5
    candidate_num = 6
    distances = {}
    tmp_parameters = {}
    for idx, parameter in dict_parameters.items():
        distance = []
        for _idx, _parameter in dict_parameters.items():
            dis = [torch.norm((parameter[name].data - _parameter[name].data).float())**2 for name in parameter.keys()]
            distance.append(sum(dis))
            tmp_parameters[idx] = parameter
        # distance = sum(torch.Tensor(distance).float())
        distance.sort()
        distances[idx] = sum(distance[:candidate_num])
    args.get_logger().info("Distances #{} on client #{}", str(distances.values()),
                           str(distances.keys()))
    sorted_distance = dict(sorted(distances.items(), key=lambda item: item[1]))
    candidate_parameters = [tmp_parameters[idx] for idx in sorted_distance.keys()][:multi_krum]
    args.get_logger().info("Averaging parameters on client #{}", str(list(sorted_distance.keys())[:multi_krum]))
    new_params = {}
    for name in candidate_parameters[0].keys():
        new_params[name] = sum([param[name].data for param in candidate_parameters]) / len(candidate_parameters)

    return new_params


def krum_nn_parameters(dict_parameters, args):
    """
    krum passed parameters.

    :param dict_parameters: nn model named parameters with client index
    :type parameters: list
    """
    from torch.nn import functional as F
    import numpy as np

    # candidate_num = args.get_num_workers() - int(args.get_num_poisoned_workers()/2) -2
    args.get_logger().info("Averaging parameters on krum")

    candidate_num = 6
    distances = {}
    tmp_parameters = {}
    for idx, parameter in dict_parameters.items():
        distance = []
        for _idx, _parameter in dict_parameters.items():
            dis = [torch.norm((parameter[name].data - _parameter[name].data).float())**2 for name in parameter.keys()]
            distance.append(sum(dis))
            tmp_parameters[idx] = parameter
        # distance = sum(torch.Tensor(distance).float())
        distance.sort()
        distances[idx] = sum(distance[:candidate_num])
    args.get_logger().info("Distances #{} on client #{}", str(distances.values()),
                           str(distances.keys()))
    sorted_distance = dict(sorted(distances.items(), key=lambda item: item[1]))
    candidate_parameters = [tmp_parameters[idx] for idx in sorted_distance.keys()][:1]
    args.get_logger().info("Averaging parameters on client #{}", str(list(sorted_distance.keys())[:1]))

    new_params = {}
    for name in candidate_parameters[0].keys():
        new_params[name] = sum([param[name].data for param in candidate_parameters]) / len(candidate_parameters)

    return new_params

def bulyan_nn_parameters(dict_parameters, args):
    """
    bulyan passed parameters.

    :param dict_parameters: nn model named parameters with client index
    :type parameters: list
    """
    args.get_logger().info("Averaging parameters on bulyan")
    multi_krum = 5
    candidate_num = 6
    distances = {}
    tmp_parameters = {}
    for idx, parameter in dict_parameters.items():
        distance = []
        for _idx, _parameter in dict_parameters.items():
            dis = [torch.norm((parameter[name].data - _parameter[name].data).float())**2 for name in parameter.keys()]
            distance.append(sum(dis))
            tmp_parameters[idx] = parameter
        # distance = sum(torch.Tensor(distance).float())
        distance.sort()
        distances[idx] = sum(distance[:candidate_num])
    args.get_logger().info("Distances #{} on client #{}", str(distances.values()),
                           str(distances.keys()))
    sorted_distance = dict(sorted(distances.items(), key=lambda item: item[1]))
    candidate_parameters = [tmp_parameters[idx] for idx in sorted_distance.keys()][1:multi_krum-1]
    args.get_logger().info("Averaging parameters on client #{}", str(list(sorted_distance.keys())[1:multi_krum-1]))
    new_params = {}
    for name in candidate_parameters[0].keys():
        new_params[name] = sum([param[name].data for param in candidate_parameters]) / len(candidate_parameters)

    return new_params

def trmean_nn_parameters(parameters, args):
    """
    Trimmed mean of passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    args.get_logger().info("Averaging parameters on trimmed mean")

    new_params = {}
    for name in parameters[0].keys():
        tmp = []
        for param in parameters:
            tmp.append(param[name].data)
        max_data = torch.max(torch.stack(tmp), 0)[0]
        min_data = torch.min(torch.stack(tmp), 0)[0]
        new_params[name] = sum([param[name].data for param in parameters])
        new_params[name] -= (max_data+min_data)
        new_params[name] /= (len(parameters)-2)

    return new_params


def median_nn_parameters(parameters, args):
    """
    median of passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    args.get_logger().info("Averaging parameters on median")

    new_params = {}
    for name in parameters[0].keys():
        tmp = []
        for param in parameters:
            tmp.append(param[name].data)
        median_data = torch.median(torch.stack(tmp), 0)[0]
        new_params[name] = median_data

    return new_params