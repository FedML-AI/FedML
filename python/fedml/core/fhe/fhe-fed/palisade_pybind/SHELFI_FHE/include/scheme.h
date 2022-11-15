#ifndef SCHEME_H
#define SCHEME_H

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <omp.h>

using namespace std;
namespace py = pybind11;

class Scheme {

private:
    string scheme;

public:
    Scheme(string scheme) : scheme(scheme){};

    virtual void loadCryptoParams() = 0;
    virtual int genCryptoContextAndKeyGen() = 0;
    virtual py::bytes encrypt(py::array_t<double> data_array) = 0;
    virtual string encrypt_cpp(vector<double> learner_Data) = 0;
    virtual py::bytes computeWeightedAverage(py::list learner_data, py::list scaling_factors) = 0;
    virtual string computeWeightedAverage_cpp(vector<string> learners_Data, vector<float> scalingFactors) = 0;
    virtual py::array_t<double> decrypt( string learner_data, unsigned long int data_dimensions) = 0;
    virtual vector<double> decrypt_cpp(string learner_Data, unsigned long int data_dimesions) = 0;
    virtual ~Scheme() = default;
};

#endif //SCHEME_H