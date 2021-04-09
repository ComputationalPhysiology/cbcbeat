cpp_stimulus = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/function/Constant.h>

class Stimulus : public dolfin::Expression
{
public:

  std::shared_ptr<dolfin::MeshFunction<std::size_t> > cell_data;
  std::shared_ptr<dolfin::Constant> t;

  Stimulus() : dolfin::Expression()
  {
  }

  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& c) const
  {
    assert(cell_data);
    assert(t);

    double t_value = *t;

    switch ((*cell_data)[c.index])
    {
    case 0:
      values[0] = 0.0;
      break;
    case 1:
      if (t_value <= duration)
        values[0] = amplitude;
      else
        values[0] = 0.0;
      break;
    default:
      values[0] = 0.0;
    }
  }
  double amplitude;
  double duration;
};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<Stimulus, std::shared_ptr<Stimulus>, dolfin::Expression>
    (m, "Stimulus")
    .def(py::init<>())
    .def_readwrite("cell_data", &Stimulus::cell_data)
    .def_readwrite("t", &Stimulus::t)
    .def_readwrite("duration", &Stimulus::duration)
    .def_readwrite("amplitude", &Stimulus::amplitude);
}


"""


debug_stimulus = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/function/Constant.h>

class Stimulus : public dolfin::Expression
{
public:

  std::shared_ptr<dolfin::MeshFunction<std::size_t> > cell_data;
  std::shared_ptr<dolfin::Constant> t;

  Stimulus() : dolfin::Expression()
  {
  }

  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& c) const
  {
    assert(cell_data);
    assert(t);

    double t_value = *t;

    switch ((*cell_data)[c.index])
    {
    case 0:
      values[0] = 0.0;
      break;
    case 1:
      if (t_value <= duration)
        values[0] = amplitude;
      else
        values[0] = 0.0;
      break;
    default:
      values[0] = 0.0;
    }
  }
  double amplitude;
  double duration;
};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<Stimulus, std::shared_ptr<Stimulus>, dolfin::Expression>
    (m, "Stimulus")
    .def(py::init<>())
    .def_readwrite("cell_data", &Stimulus::cell_data)
    .def_readwrite("t", &Stimulus::t)
    .def_readwrite("duration", &Stimulus::duration)
    .def_readwrite("amplitude", &Stimulus::amplitude);
}


"""

