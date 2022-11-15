#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include "scheme.h"


#include <iostream>
#include <fstream>
#include <sstream>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "palisade.h"
#include "pubkeylp-ser.h"
#include "scheme/ckks/ckks-ser.h"

using namespace std;
using namespace lbcrypto;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
namespace py = pybind11;

class CKKS : public Scheme {

private:

    uint batchSize;
    uint scaleFactorBits;
    std::string cryptodir;

    CryptoContext<DCRTPoly> cc;
    LPPublicKey<DCRTPoly> pk;
    LPPrivateKey<DCRTPoly> sk;

public:
    CKKS(string scheme, uint batchSize, uint scaleFactorBits, string cryptodir);

    virtual int genCryptoContextAndKeyGen();
    virtual void loadCryptoParams();

    virtual py::bytes encrypt(py::array_t<double> data_array);
    virtual string encrypt_cpp(vector<double> learner_Data);

    virtual py::bytes computeWeightedAverage(py::list learner_data, py::list scaling_factors);
    virtual string computeWeightedAverage_cpp(vector<string> learners_Data, vector<float> scalingFactors);

    virtual py::array_t<double> decrypt(string learner_data, unsigned long int data_dimensions);
    virtual vector<double> decrypt_cpp(string learner_Data, unsigned long int data_dimesions);

    
};