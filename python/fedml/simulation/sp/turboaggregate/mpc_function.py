import numpy as np


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
    # print(_num,_den,_inv)
    return np.mod(np.int64(_num) * np.int64(_inv), _p)


def PI(vals, p):  # upper-case PI -- product of inputs
    accum = 1

    for v in vals:
        tmp = np.mod(v, p)
        accum = np.mod(accum * tmp, p)
    return accum


def gen_Lagrange_coeffs(alpha_s, beta_s, p, is_K1=0):
    if is_K1 == 1:
        num_alpha = 1
    else:
        num_alpha = len(alpha_s)
    U = np.zeros((num_alpha, len(beta_s)), dtype="int64")
    #         U = [[0 for col in range(len(beta_s))] for row in range(len(alpha_s))]
    # print(alpha_s)
    # print(beta_s)
    for i in range(num_alpha):
        for j in range(len(beta_s)):
            cur_beta = beta_s[j]

            den = PI([cur_beta - o for o in beta_s if cur_beta != o], p)
            num = PI([alpha_s[i] - o for o in beta_s if cur_beta != o], p)
            U[i][j] = divmod(num, den, p)
            # for debugging
            # print(i,j,cur_beta,alpha_s[i])
            # print(test)
            # print(den,num)
    return U.astype("int64")


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
    # RT_LCC = f_deg * (K + T - 1) + 1

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


def LCC_encoding_with_points(X, alpha_s, beta_s, p):
    m, d = np.shape(X)

    # print alpha_s
    # print beta_s

    # for debugging LCC Enc & Dec
    # beta_s = np.concatenate((alpha_s, beta_s))
    # print beta_s

    U = gen_Lagrange_coeffs(beta_s, alpha_s, p).astype("int")
    # print U

    X_LCC = np.zeros((len(beta_s), d), dtype="int")
    for i in range(len(beta_s)):
        X_LCC[i, :] = np.dot(np.reshape(U[i, :], (1, len(alpha_s))), X)
    # print X
    # print np.mod(X_LCC, p)

    return np.mod(X_LCC, p)


def LCC_decoding_with_points(f_eval, eval_points, target_points, p):
    alpha_s_eval = eval_points
    beta_s = target_points

    U_dec = gen_Lagrange_coeffs(beta_s, alpha_s_eval, p)

    # print U_dec

    f_recon = np.mod((U_dec).dot(f_eval), p)
    # print f_recon

    return f_recon


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
