#include "LightSecAgg.h"


int LightSecAgg::modInverse(int a, int p) {
    int m = p;
    int y = 0, x = 1;
    int q = 0;

    while (a > 1) {
        if (m != 0) {
            q = a / m;
            int t = m;
            m = a % m, a = t;
            t = y;
            y = x - q * y;
            x = t;
        } else {
            q = 0;
            a = 0;
            int t = y;
            y = x - q * y;
            x = t;
        }
        if (x < 0) {
            x = x + p;
        }
    }
    x = x % p;
    if (x < 0)
        x = x + p;
    return x;
}

int LightSecAgg::modDivide(int num, int den, int p) {
    num = num % p;
    den = den % p;
    int inv = modInverse(den, p);
    int c = (inv * num) % p;
    return c;
}


int LightSecAgg::PI(std::vector<int> vals, int p) {
    int accum = 1;
    for (auto v: vals) {
        if (v < 0)
            v = v + p;
        int tmp = v % p;
        accum = accum * tmp % p;
    }
    return accum;
}


std::vector <std::vector<int>> LightSecAgg::gen_Lagrange_coeffs(std::vector<int> const &alpha_s,
                                                                std::vector<int> const &beta_s,
                                                                int p, int is_K1) {
    int num_alpha = (is_K1 == 1) ? 1 : alpha_s.size();
    std::vector <std::vector<int>> U(num_alpha, std::vector<int>(beta_s.size(), 0));

    std::vector<int> w(beta_s.size(), 0);
    for (int j = 0; j < beta_s.size(); j++) {
        int cur_beta = beta_s[j];
        std::vector<int> val;
        for (auto o: beta_s) {
            if (cur_beta != o)
                val.push_back(cur_beta - o);
        }
        int den = PI(val, p);
        w[j] = den;
    }

    std::vector<int> l(num_alpha, 0);
    for (int i = 0; i < num_alpha; i++) {
        std::vector<int> val;
        for (auto o: beta_s) {
            val.push_back(alpha_s[i] - o);
        }
        l[i] = PI(val, p);
    }

    for (int j = 0; j < beta_s.size(); j++) {
        for (int i = 0; i < num_alpha; i++) {
            int tmp = alpha_s[i] - beta_s[j];
            if (tmp < 0)
                tmp = tmp + p;
            int den = (tmp % p) * w[j] % p;
            // int den = ((alpha_s[i] - beta_s[j]) % p) * w[j] % p;
            U[i][j] = modDivide(l[i], den, p);
        }
    }

    return U;
}

std::vector <std::vector<float>> LightSecAgg::LCC_encoding_with_points(std::vector <std::vector<float>> const &X,
                                                                       std::vector<int> const &alpha_s,
                                                                       std::vector<int> const &beta_s, int p) {
    int m = X.size();
    int d = X[0].size();
    auto U = gen_Lagrange_coeffs(beta_s, alpha_s, p);
    std::vector <std::vector<float>> X_LCC(beta_s.size(), std::vector<float>(d, 0.0));
    for (int i = 0; i < U.size(); i++) {
        for (int j = 0; j < d; j++) {
            X_LCC[i][j] = 0;
            for (int k = 0; k < U[0].size(); k++) {
                X_LCC[i][j] += U[i][k] * X[k][j];
            }
            X_LCC[i][j] = std::fmod(X_LCC[i][j], p);
        }
    }

    return X_LCC;
}

std::vector <std::vector<float>> LightSecAgg::LCC_decoding_with_points(std::vector <std::vector<float>> f_eval,
                                                                       std::vector<int> eval_points,
                                                                       std::vector<int> target_points, int p) {
    auto alpha_s_eval = eval_points;
    auto beta_s = target_points;
    auto U_dec = gen_Lagrange_coeffs(beta_s, alpha_s_eval, p);

    std::vector <std::vector<float>> f_recon(U_dec.size(), std::vector<float>(f_eval[0].size(), 0.0));
    for (int i = 0; i < U_dec.size(); i++) {
        for (int j = 0; j < f_eval[0].size(); j++) {
            f_recon[i][j] = 0;
            for (int k = 0; k < U_dec[0].size(); k++) {
                f_recon[i][j] += U_dec[i][k] * f_eval[k][j];
            }
            f_recon[i][j] = std::fmod(f_recon[i][j], p);
        }
    }
    return f_recon;

}