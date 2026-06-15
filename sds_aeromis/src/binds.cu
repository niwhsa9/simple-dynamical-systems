#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "tensor.cuh"

using namespace sds;
namespace py = pybind11;

template <typename T, int dim>
py::array_t<T> tensor_to_pyarray_copy(Tensor<T, dim>& t)
{
  Tensor<T, dim>::sync();

  std::vector<ssize_t> shape(dim);
  std::vector<ssize_t> strides(dim);
  for (int i = 0; i < dim; ++i)
  {
    shape[i] = t.shape(i);
    strides[i] = t.stride(i) * sizeof(T);
  }

  // Allocates and copies — numpy owns the result, Tensor can be freed safely
  py::array_t<T> arr(shape, strides);
  std::memcpy(arr.mutable_data(), t.data(), t.numel() * sizeof(T));
  return arr;
}

template <DynamicalSystem Sys, Integrator Int>
void bind_helper(py::module& m, const std::string& name)
{
  py::class_<Sys>(m, name.c_str())
      .def(py::init<>())
      .def("dynamics", &Sys::dynamics)
      .def("get_n_x", &Sys::get_n_x)
      .def("get_n_u", &Sys::get_n_u);
}
