#ifndef LIGHTSECAGG_CPP_MPC_FUNC_H
#define LIGHTSECAGG_CPP_MPC_FUNC_H

#include <iostream>
#include <vector>
#include <math.h>
#include <numeric>
#include <random>


class LightSecAgg {

public:
    std::vector<std::vector<float>> LCC_encoding_with_points(std::vector<std::vector<float>> const &X,
                                                              std::vector<int> const &alpha_s,
                                                              std::vector<int> const &beta_s, int p);


    std::vector<std::vector<float>> LCC_decoding_with_points(std::vector<std::vector<float>> f_eval,
                                                             std::vector<int> eval_points,
                                                             std::vector<int> target_points, int p);
private:
    int modInverse(int a, int p);

    int modDivide(int num, int den, int p);

    int PI(std::vector<int> vals, int p);

    std::vector<std::vector<int>> gen_Lagrange_coeffs(std::vector<int> const &alpha_s,
                                                       std::vector<int> const &beta_s,
                                                       int p, int is_K1 = 0);
};


#endif //LIGHTSECAGG_CPP_MPC_FUNC_H
