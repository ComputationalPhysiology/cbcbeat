cpp_stimulus = """
class Stimulus : public Expression
{
public:

  std::shared_ptr<MeshFunction<std::size_t> > cell_data;
  std::shared_ptr<Constant> t;

  Stimulus() : Expression(), amplitude(0), duration(0)
  {
  }

  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& c) const
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
};"""


debug_stimulus = """
class Stimulus : public Expression
{
public:

  std::shared_ptr<Constant> t;

  Stimulus() : Expression(), amplitude(0), duration(0)
  {
  }

  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& c) const
  {
    assert(t);
    double t_value = *t;

    if (x[0] <= 1.0/520)
    {
      values[0] = 0.0;
    }
    else
    {
      if (t_value <= duration)
        values[0] = amplitude;
      else
        values[0] = 0.0;
    }
  }
  double amplitude;
  double duration;
};"""

