import torch

def trmean_nn_parameters(parameters):
    """
    Trimmed mean of passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """

    new_params = {}
    for name in parameters[0].keys():
        tmp = []
        for param in parameters:
            tmp.append(param[name].data)
        max_data = torch.max(torch.stack(tmp), 0)[0]
        min_data = torch.min(torch.stack(tmp), 0)[0]
        new_params[name] = sum([param[name].data for param in parameters])-(max_data+min_data) / (len(parameters)-2)
    print(new_params)

    return new_params

parameter1 = {"a": torch.tensor([[1,2,3],[2,4,6],[3,6,1], [2,2,5]]), "b": torch.tensor([[1,2,9],[2,2,6], [9,0,1], [2,0,1]])}
parameter2 = {"a": torch.tensor([[1,0,3],[2,4,6],[1,6,1], [7,2,5]]), "b": torch.tensor([[4,1,9],[2,2,6], [0,0,1], [3,2,1]])}
parameters = []
parameters.append(parameter1)
parameters.append(parameter2)
print(trmean_nn_parameters(parameters))