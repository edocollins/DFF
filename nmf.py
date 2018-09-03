import torch, math, time

EPSILON = 1e-7

# NMF by multiplictive updates
def NMF(V, k, W=None, H=None, random_seed=None, max_iter=200, tol=1e-4, cuda=True, verbose=False):

    if verbose:
        start_time = time.time()

    scale = math.sqrt(V.mean() / k)

    if random_seed is not None:
        if cuda:
            current_random_seed = torch.cuda.initial_seed()
            torch.cuda.manual_seed(random_seed)
        else:
            current_random_seed = torch.initial_seed()
            torch.manual_seed(random_seed)

    if W is None:
        if cuda:
            W = torch.cuda.FloatTensor(V.size(0), k).normal_()
        else:
            W = torch.randn(V.size(0), k)
        W *= scale

    update_H = True
    if H is None:
        if cuda:
            H = torch.cuda.FloatTensor(k, V.size(1)).normal_()
        else:
            H = torch.randn(k, V.size(1))
        H *= scale
    else:
        update_H = False

    if random_seed is not None:
        if cuda:
            torch.cuda.manual_seed(current_random_seed)
        else:
            torch.manual_seed(current_random_seed)

    W = torch.abs(W)
    H = torch.abs(H)

    error_at_init = approximation_error(V, W, H, square_root=True)
    previous_error = error_at_init

    VH = None
    HH = None
    for n_iter in range(max_iter):
        W, H, VH, HH = multiplicative_update_step(V, W, H, update_H=update_H, VH = VH, HH = HH)
        if tol > 0 and n_iter % 10 == 0:
            error = approximation_error(V, W, H, square_root=True)

            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error
    if verbose:
        print('Exited after {} iterations. Total time: {} seconds'.format(n_iter+1, time.time()-start_time))
    return W, H


def multiplicative_update_step(V, W, H, update_H=True, VH=None, HH=None):
    # update operation for W
    if VH is None:
        assert HH is None
        Ht = torch.t(H)
        VH = torch.mm(V, Ht)
        HH = torch.mm(H, Ht)

    WHH = torch.mm(W, HH)
    WHH[WHH == 0] = EPSILON
    W *= VH / WHH

    if update_H:
        # update operation for H (after updating W)
        Wt = torch.t(W)
        WV = torch.mm(Wt, V)
        WWH = torch.mm(torch.mm(Wt, W), H)
        WWH[WWH == 0] = EPSILON
        H *= WV / WWH
        VH, HH = None, None

    return W, H, VH, HH


# NMF objective
def approximation_error(V, W, H, square_root=True):
    # Frobenius norm
    return torch.norm(V - torch.mm(W, H))