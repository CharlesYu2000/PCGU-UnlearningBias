def _get_all_indices(shape, dim_to_agg): 
    dims = len(shape)

    def _get_all_indices_helper(curr_inds, permute_from_dim):
        if permute_from_dim==dims: 
            if dim_to_agg==-1: 
                # remove the range
                return [curr_inds[:-1]]
            else: 
                return [curr_inds]
        else: 
            # put all values
            ind_vals = range(shape[permute_from_dim])

            if permute_from_dim==dims+dim_to_agg: 
                # make the range as the only possible value
                ind_vals = [ind_vals] 

            # recurse
            all_permutations = []
            for outer_perm_val in ind_vals: 
                curr_inds = curr_inds[:permute_from_dim] + (outer_perm_val,) + curr_inds[permute_from_dim+1:]
                all_permutations.extend(_get_all_indices_helper(curr_inds, permute_from_dim+1))
            return all_permutations

    init_inds = (0,)*dims
    return _get_all_indices_helper(init_inds, 0)

def create_param_partition(params, dim_to_agg=-1): 
    '''
    dim_to_agg=-1 for "input" aggregation, dim_to_agg=-2 for "output" aggregation
    '''

    assert dim_to_agg in {-1,-2}, 'can only guarantee aggregation for dims -1, -2 (other dims might not exist). Code works for other dims if they exist..'

    param_partition = []
    for param_name, param in params.items(): 
        if not param.requires_grad: 
            continue
        dims = len(param.shape)
        if dims>1:
            for indices in _get_all_indices(param.shape, dim_to_agg=dim_to_agg): # split into vectors based on dim_to_agg
                param_partition.append((param_name, indices))
        elif dims==1: # if already vector, no aggregation is done
            param_partition.append((param_name, None))
        else: # dims==0, should never happen, means there are params that are single values, so cosine sim would just be 1,0(?),-1
            raise ValueError(f'param {param_name} has dimension of 0, shape is: {param.shape}')
    return param_partition