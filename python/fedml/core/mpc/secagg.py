import copy
import logging

import numpy as np
import torch


def modular_inv(a, p):
    x, y, m = 1, 0, p
    while a > 1:
        q = a // m
        t = m

        m = np.mod(a, m)
        a = t
        t = y

        y, x = x - np.int64(q) * np.int64(y), t

        if x < 0:
            x = np.mod(x, p)
    return np.mod(x, p)


def divmod(_num, _den, _p):
    # compute num / den modulo prime p
    _num = np.mod(_num, _p)
    _den = np.mod(_den, _p)
    _inv = modular_inv(_den, _p)
    return np.mod(np.int64(_num) * np.int64(_inv), _p)


def PI(vals, p):  # upper-case PI -- product of inputs
    accum = np.int64(1)
    for v in vals:
        tmp = np.mod(v, p)
        accum = np.mod(accum * tmp, p)
    return accum


def LCC_encoding_with_points(X, alpha_s, beta_s, p):
    m, d = np.shape(X)
    U = gen_Lagrange_coeffs(beta_s, alpha_s, p).astype("int64")
    X_LCC = np.zeros((len(beta_s), d), dtype="int64")
    for i in range(len(beta_s)):
        X_LCC[i, :] = np.dot(np.reshape(U[i, :], (1, len(alpha_s))), X)
    return np.mod(X_LCC, p)


def LCC_decoding_with_points(f_eval, eval_points, target_points, p):
    alpha_s_eval = eval_points
    beta_s = target_points
    U_dec = gen_Lagrange_coeffs(beta_s, alpha_s_eval, p)
    f_recon = np.mod((U_dec).dot(f_eval), p)

    return f_recon


def gen_Lagrange_coeffs(alpha_s, beta_s, p, is_K1=0):
    if is_K1 == 1:
        num_alpha = 1
    else:
        num_alpha = len(alpha_s)
    U = np.zeros((num_alpha, len(beta_s)), dtype="int64")

    w = np.zeros((len(beta_s)), dtype="int64")
    for j in range(len(beta_s)):
        cur_beta = beta_s[j]
        den = PI([cur_beta - o for o in beta_s if cur_beta != o], p)
        w[j] = den

    l = np.zeros((num_alpha), dtype="int64")
    for i in range(num_alpha):
        l[i] = PI([alpha_s[i] - o for o in beta_s], p)

    for j in range(len(beta_s)):
        for i in range(num_alpha):
            den = np.mod(np.mod(alpha_s[i] - beta_s[j], p) * w[j], p)
            U[i][j] = divmod(l[i], den, p)
    return U.astype("int64")


def model_masking(weights_finite, dimensions, local_mask, prime_number):
    pos = 0
    reshaped_local_mask = local_mask.reshape((local_mask.shape[0], 1))
    for i, k in enumerate(weights_finite):

        # if k== "module.conv1.weight":
        #     logging.info("before mask")
        #     logging.info(weights_finite[k][0])
        # if i == 0:
        #     logging.info("%^%^&%^&%^&%^&^&%^&")
        #     logging.info("Before masking")
        #     logging.info(weights_finite[k][0])
        tmp = weights_finite[k]
        cur_shape = tmp.shape
        d = dimensions[i]
        cur_mask = reshaped_local_mask[pos : pos + d, :]
        cur_mask = np.reshape(cur_mask, cur_shape)
        weights_finite[k] += cur_mask
        weights_finite[k] = np.mod(weights_finite[k], prime_number)
        # if i == 0:
        #     logging.info("Mask")
        #     logging.info(cur_mask[0])
        #     logging.info("After masking")
        #     logging.info(weights_finite[k][0])
        #     logging.info("%^%^&%^&%^&%^&^&%^&")

        # if k== "module.conv1.weight":
        #     logging.info("Mask")
        #     logging.info(cur_mask[0])
        #     logging.info("After Mask")
        #     logging.info(weights_finite[k][0])
        pos += d
    return weights_finite


def mask_encoding(
    total_dimension, num_clients, targeted_number_active_clients, privacy_guarantee, prime_number, local_mask
):
    d = total_dimension
    N = num_clients
    U = targeted_number_active_clients
    T = privacy_guarantee
    p = prime_number

    beta_s = np.array(range(N)) + 1
    alpha_s = np.array(range(U)) + (N + 1)

    # n_i = np.random.randint(p, size=(T * d // (U - T), 1))
    n_i = np.zeros((T * d // (U - T), 1)).astype("int64")

    LCC_in = np.concatenate([local_mask, n_i], axis=0)
    LCC_in = np.reshape(LCC_in, (U, d // (U - T)))
    encoded_mask_set = LCC_encoding_with_points(LCC_in, alpha_s, beta_s, p).astype("int64")

    return encoded_mask_set


def compute_aggregate_encoded_mask(encoded_mask_dict, p, active_clients):
    aggregate_encoded_mask = np.zeros((np.shape(encoded_mask_dict[0])))
    for client_id in active_clients:
        aggregate_encoded_mask += encoded_mask_dict[client_id]
    aggregate_encoded_mask = np.mod(aggregate_encoded_mask, p).astype("int")
    return aggregate_encoded_mask


def aggregate_models_in_finite(weights_finite, prime_number):
    """
    weights_finite : array of state_dict()
    prime_number   : size of the finite field
    """
    w_sum = copy.deepcopy(weights_finite[0])

    for key in w_sum.keys():

        for i in range(1, len(weights_finite)):
            w_sum[key] += weights_finite[i][key]
            w_sum[key] = np.mod(w_sum[key], prime_number)

    return w_sum


def BGW_encoding(X, N, T, p):
    m = len(X)
    d = len(X[0])

    alpha_s_range = range(1, N + 1)
    alpha_s = np.array(np.int64(np.mod(alpha_s_range, p)))
    X_BGW = np.zeros((N, m, d), dtype="int64")
    R = np.random.randint(p, size=(T + 1, m, d))
    R[0, :, :] = np.mod(X, p)

    for i in range(N):
        for t in range(T + 1):
            X_BGW[i, :, :] = np.mod(X_BGW[i, :, :] + R[t, :, :] * (alpha_s[i] ** t), p)
    return X_BGW


def gen_BGW_lambda_s(alpha_s, p):
    lambda_s = np.zeros((1, len(alpha_s)), dtype="int64")

    for i in range(len(alpha_s)):
        cur_alpha = alpha_s[i]

        den = PI([cur_alpha - o for o in alpha_s if cur_alpha != o], p)
        num = PI([0 - o for o in alpha_s if cur_alpha != o], p)
        lambda_s[0][i] = divmod(num, den, p)
    return lambda_s.astype("int64")


def BGW_decoding(f_eval, worker_idx, p):  # decode the output from T+1 evaluation points
    # f_eval     : [RT X d ]
    # worker_idx : [ 1 X RT]
    # output     : [ 1 X d ]

    # t0 = time.time()
    max = np.max(worker_idx) + 2
    alpha_s_range = range(1, max)
    alpha_s = np.array(np.int64(np.mod(alpha_s_range, p)))
    alpha_s_eval = [alpha_s[i] for i in worker_idx]
    # t1 = time.time()
    # print(alpha_s_eval)
    lambda_s = gen_BGW_lambda_s(alpha_s_eval, p).astype("int64")
    # t2 = time.time()
    # print(lambda_s.shape)
    f_recon = np.mod(np.dot(lambda_s, f_eval), p)
    # t3 = time.time()
    # print 'time info for BGW_dec', t1-t0, t2-t1, t3-t2
    return f_recon


def LCC_encoding(X, N, K, T, p):
    m = len(X)
    d = len(X[0])
    # print(m,d,m//K)
    X_sub = np.zeros((K + T, m // K, d), dtype="int64")
    for i in range(K):
        X_sub[i] = X[i * m // K : (i + 1) * m // K :]
    for i in range(K, K + T):
        X_sub[i] = np.random.randint(p, size=(m // K, d))

    n_beta = K + T
    stt_b, stt_a = -int(np.floor(n_beta / 2)), -int(np.floor(N / 2))
    beta_s, alpha_s = range(stt_b, stt_b + n_beta), range(stt_a, stt_a + N)
    alpha_s = np.array(np.mod(alpha_s, p)).astype("int64")
    beta_s = np.array(np.mod(beta_s, p)).astype("int64")

    U = gen_Lagrange_coeffs(alpha_s, beta_s, p)
    # print U

    X_LCC = np.zeros((N, m // K, d), dtype="int64")
    for i in range(N):
        for j in range(K + T):
            X_LCC[i, :, :] = np.mod(X_LCC[i, :, :] + np.mod(U[i][j] * X_sub[j, :, :], p), p)
    return X_LCC


def LCC_encoding_w_Random(X, R_, N, K, T, p):
    m = len(X)
    d = len(X[0])
    # print(m,d,m//K)
    X_sub = np.zeros((K + T, m // K, d), dtype="int64")
    for i in range(K):
        X_sub[i] = X[i * m // K : (i + 1) * m // K :]
    for i in range(K, K + T):
        X_sub[i] = R_[i - K, :, :].astype("int64")

    n_beta = K + T
    stt_b, stt_a = -int(np.floor(n_beta / 2)), -int(np.floor(N / 2))
    beta_s, alpha_s = range(stt_b, stt_b + n_beta), range(stt_a, stt_a + N)

    alpha_s = np.array(np.mod(alpha_s, p)).astype("int64")
    beta_s = np.array(np.mod(beta_s, p)).astype("int64")

    # alpha_s = np.int64(np.mod(alpha_s,p))
    # beta_s = np.int64(np.mod(beta_s,p))

    U = gen_Lagrange_coeffs(alpha_s, beta_s, p)
    # print U

    X_LCC = np.zeros((N, m // K, d), dtype="int64")
    for i in range(N):
        for j in range(K + T):
            X_LCC[i, :, :] = np.mod(X_LCC[i, :, :] + np.mod(U[i][j] * X_sub[j, :, :], p), p)
    return X_LCC


def LCC_encoding_w_Random_partial(X, R_, N, K, T, p, worker_idx):
    m = len(X)
    d = len(X[0])
    # print(m,d,m//K)
    X_sub = np.zeros((K + T, m // K, d), dtype="int64")
    for i in range(K):
        X_sub[i] = X[i * m // K : (i + 1) * m // K :]
    for i in range(K, K + T):
        X_sub[i] = R_[i - K, :, :].astype("int64")

    n_beta = K + T
    stt_b, stt_a = -int(np.floor(n_beta / 2)), -int(np.floor(N / 2))
    beta_s, alpha_s = range(stt_b, stt_b + n_beta), range(stt_a, stt_a + N)
    alpha_s = np.array(np.mod(alpha_s, p)).astype("int64")
    beta_s = np.array(np.mod(beta_s, p)).astype("int64")
    alpha_s_eval = [alpha_s[i] for i in worker_idx]

    U = gen_Lagrange_coeffs(alpha_s_eval, beta_s, p)
    # print U

    N_out = U.shape[0]
    X_LCC = np.zeros((N_out, m // K, d), dtype="int64")
    for i in range(N_out):
        for j in range(K + T):
            X_LCC[i, :, :] = np.mod(X_LCC[i, :, :] + np.mod(U[i][j] * X_sub[j, :, :], p), p)
    return X_LCC


def LCC_decoding(f_eval, f_deg, N, K, T, worker_idx, p):
    RT_LCC = f_deg * (K + T - 1) + 1

    n_beta = K  # +T
    stt_b, stt_a = -int(np.floor(n_beta / 2)), -int(np.floor(N / 2))
    beta_s, alpha_s = range(stt_b, stt_b + n_beta), range(stt_a, stt_a + N)
    alpha_s = np.array(np.mod(alpha_s, p)).astype("int64")
    beta_s = np.array(np.mod(beta_s, p)).astype("int64")
    alpha_s_eval = [alpha_s[i] for i in worker_idx]

    U_dec = gen_Lagrange_coeffs(beta_s, alpha_s_eval, p)

    # print U_dec

    f_recon = np.mod((U_dec).dot(f_eval), p)

    return f_recon.astype("int64")


def Gen_Additive_SS(d, n_out, p):
    # x_model should be one dimension

    temp = np.random.randint(0, p, size=(n_out - 1, d))
    # print temp

    last_row = np.reshape(np.mod(-np.sum(temp, axis=0), p), (1, d))
    Additive_SS = np.concatenate((temp, last_row), axis=0)
    # print np.mod(np.sum(Additive_SS,axis=0),p)

    return Additive_SS


def my_pk_gen(my_sk, p, g):
    # print 'my_pk_gen option: g=',g
    if g == 0:
        return my_sk
    else:
        return np.mod(g ** my_sk, p)


def my_key_agreement(my_sk, u_pk, p, g):
    if g == 0:
        return np.mod(my_sk * u_pk, p)
    else:
        return np.mod(u_pk ** my_sk, p)


def my_q(X, q_bit, p):
    X_int = np.round(X * (2 ** q_bit))
    is_negative = (abs(np.sign(X_int)) - np.sign(X_int)) / 2
    out = X_int + p * is_negative
    return out.astype("int64")


def transform_tensor_to_finite(model_params, p, q_bits):
    for k in model_params.keys():
        tmp = np.array(model_params[k])
        tmp_finite = my_q(tmp, q_bits, p)
        model_params[k] = tmp_finite
    return model_params


def my_q_inv(X_q, q_bit, p):
    flag = X_q - (p - 1) / 2
    is_negative = (abs(np.sign(flag)) + np.sign(flag)) / 2
    X_q = X_q - p * is_negative
    return X_q.astype(float) / (2 ** q_bit)


def transform_finite_to_tensor(model_params, p, q_bits):
    for k in model_params.keys():
        tmp = np.array(model_params[k])
        tmp_real = my_q_inv(tmp, q_bits, p)
        """
        Handle two types of numpy variables, array and float
        0 - Wed, 13 Oct 2021 07:50:59 utils.py[line:33] 
        DEBUG tmp_real = [4.20669909e+07 3.95378213e+08 3.20326493e+08 2.69357568e+08
         1.50241777e+08 4.17396229e+08 4.70202170e+07 2.97959914e+08
         3.03893524e+08 5.43255322e+07 4.22816876e+08 3.57158791e+06
         1.56766933e+08 2.36449153e+08 6.66361822e+07 1.66616215e+08]
        0 - Wed, 13 Oct 2021 07:50:59 utils.py[line:33] DEBUG tmp_real = 256812209.4375
        """
        # logging.debug("tmp_real = {}".format(tmp_real))
        tmp_real = torch.Tensor([tmp_real]) if isinstance(tmp_real, np.floating) else torch.Tensor(tmp_real)
        model_params[k] = tmp_real
    return model_params


def model_dimension(weights):
    logging.info("Get model dimension")
    dimensions = []
    for k in weights.keys():
        tmp = weights[k].cpu().detach().numpy()
        cur_shape = tmp.shape
        _d = int(np.prod(cur_shape))
        dimensions.append(_d)
    total_dimension = sum(dimensions)
    logging.info("Dimension of model d is %d." % total_dimension)
    return dimensions, total_dimension
