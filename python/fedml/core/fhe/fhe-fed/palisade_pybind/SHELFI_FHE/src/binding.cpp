#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "scheme.h"
#include "ckks.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
namespace py = pybind11;

PYBIND11_MODULE(SHELFI_FHE, m) {

py::class_<Scheme>(m, "Scheme");

py::class_<CKKS, Scheme>(m, "CKKS")
        .def(py::init<std::string &, uint, uint, std::string &>(),
            py::arg("scheme") = py::str("ckks"),
            py::arg("batchSize") = 4096, 
            py::arg("scaleFactorBits") = 52, 
            py::arg("cryptodir") = py::str("../resources/cryptoparams/"))
      .def("loadCryptoParams", &CKKS::loadCryptoParams)
      .def("genCryptoContextAndKeyGen", &CKKS::genCryptoContextAndKeyGen)
      .def("encrypt", &CKKS::encrypt)
      .def("encrypt_cpp", &CKKS::encrypt)
      .def("decrypt", &CKKS::decrypt)
      .def("decrypt_cpp", &CKKS::decrypt)
      .def("computeWeightedAverage", &CKKS::computeWeightedAverage)
      .def("computeWeightedAverage_cpp", &CKKS::computeWeightedAverage);





m.doc() = R"pbdoc(
      Pybind11 example plugin
      -----------------------
      .. currentmodule:: cmake_example
      .. autosummary::
         :toctree: _generate
  )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

}