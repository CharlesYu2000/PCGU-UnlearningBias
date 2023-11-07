import logging

logger = logging.getLogger(__name__)

def get_params_map(model): 
    params_map = {}
    for param_name, param in model.named_parameters(): 
        params_map[param_name] = param
    return params_map

def get_all_model_grads(model, clear_grads_after=False): 
    params = list(model.named_parameters())
    param_grads = {}
    problems = []
    for i, (param_name, param) in enumerate(params): 
        if param.requires_grad: 
            try: 
                param_grad = param.grad.detach().cpu()
                # param_grad = param_grad.reshape(-1)
            except: 
                problems.append(f'{i}: {param_name}')
                param_grad = None
            param_grads[param_name] = param_grad
    if problems: 
        logger.debug(f'Problems: {problems}')
    if clear_grads_after: 
        model.zero_grad()
    return param_grads

def accumulate_grad(accumulator_grad, grad): 
    if accumulator_grad is None: 
        return grad

    for param_name, curr_grad_val in accumulator_grad.items(): 
        if (curr_grad_val is None)!=(grad[param_name] is None): 
            raise RuntimeError(f'Values not same none-ness for param {param_name}')
        if (curr_grad_val is None): 
            accumulator_grad[param_name] = grad[param_name]
        else: 
            accumulator_grad[param_name] = (curr_grad_val + grad[param_name]).detach()

    return accumulator_grad