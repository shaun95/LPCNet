// pybind11_wrapper.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "freq.h"

#define LPC_ORDER 16
namespace py = pybind11;

py::array_t<float> lpc_from_cepstrum_numpy(py::array_t<float> cepstrum_array) {
    py::buffer_info cepstrum_buf = cepstrum_array.request();

    auto lpc_array = py::array_t<float>(LPC_ORDER);

    py::buffer_info lpc_buf = lpc_array.request();

    float *cepstrum_ptr = (float *) cepstrum_buf.ptr;
    float *lpc_ptr = (float *) lpc_buf.ptr;
		lpc_from_cepstrum(lpc_ptr, cepstrum_ptr);

    return lpc_array;
}


PYBIND11_MODULE(LPCNet, m) {
    m.doc() = "pybind11 example plugin"; // Optional module docstring
    m.def("lpc_from_cepstrum", &lpc_from_cepstrum_numpy, "A function calculate lpc from cepstrum");
  }
