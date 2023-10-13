from pathlib import Path
from typing import NamedTuple, Dict, Tuple
import dolfin
import cbcbeat
import numpy as np
import ufl
import ldrb  # pip install ldrb


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

  void eval(Eigen::Ref<Eigen::VectorXd> values,
            Eigen::Ref<const Eigen::VectorXd> x,
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


def setup_general_parameters():
    # Adjust some general FEniCS related parameters
    dolfin.parameters["form_compiler"]["representation"] = "uflacs"
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True
    flags = ["-O3", "-ffast-math"]
    dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
    dolfin.parameters["form_compiler"]["quadrature_degree"] = 2


class Geometry(NamedTuple):
    heart_mesh: dolfin.Mesh
    torso_mesh: dolfin.Mesh
    heart_ffun: dolfin.MeshFunction
    torso_ffun: dolfin.MeshFunction
    heart_markers: Dict[str, Tuple[int, int]]
    torso_markers: Dict[str, Tuple[int, int]]


class Data(NamedTuple):
    geo: Geometry
    stimulation_cells: dolfin.MeshFunction
    f0: dolfin.Function
    s0: dolfin.Function
    n0: dolfin.Function


class Conductivities(NamedTuple):
    chi: float
    C_m: float
    g_el: float
    g_et: float
    g_il: float
    g_it: float
    g_el_field: dolfin.Function
    g_et_field: dolfin.Function
    g_il_field: dolfin.Function
    g_it_field: dolfin.Function
    M_i: ufl.tensors.ComponentTensor
    M_e: ufl.tensors.ComponentTensor


def setup_conductivities(data: Data):
    chi = 1400.0  # Membrane surface-to-volume ratio (1/cm)
    C_m = 1.0  # Membrane capacitance per unit area (micro F/(cm^2))

    V = dolfin.FunctionSpace(data.geo.heart_mesh, "CG", 1)

    g_el_field = dolfin.Function(V)
    g_et_field = dolfin.Function(V)
    g_il_field = dolfin.Function(V)
    g_it_field = dolfin.Function(V)

    # Extracellular:
    g_el = 6.25 / (C_m * chi)  # Fiber
    g_et = 2.36 / (C_m * chi)  # Sheet
    g_et = 2.36 / (C_m * chi)  # Cross-sheet

    # Intracellular:
    g_il = 1.74 / (C_m * chi)  # Fiber
    g_it = 0.192 / (C_m * chi)  # Sheet
    g_it = 0.192 / (C_m * chi)  # Cross-sheet
    g_el_field.vector()[:] = g_el
    g_et_field.vector()[:] = g_et
    g_il_field.vector()[:] = g_il
    g_it_field.vector()[:] = g_it

    A = dolfin.as_matrix(
        [
            [data.f0[0], data.s0[0], data.n0[0]],
            [data.f0[1], data.s0[1], data.n0[1]],
            [data.f0[2], data.s0[2], data.n0[2]],
        ]
    )
    from ufl import diag

    M_e_star = diag(dolfin.as_vector([g_el_field, g_et_field, g_et_field]))
    M_i_star = diag(dolfin.as_vector([g_il_field, g_it_field, g_it_field]))
    M_e = A * M_e_star * A.T
    M_i = A * M_i_star * A.T

    return Conductivities(
        chi=chi,
        C_m=C_m,
        g_el=g_el,
        g_et=g_et,
        g_il=g_il,
        g_it=g_it,
        g_el_field=g_el_field,
        g_et_field=g_et_field,
        g_il_field=g_il_field,
        g_it_field=g_it_field,
        M_i=M_i,
        M_e=M_e,
    )


def save_data(
    filename,
    stimulation_cells,
    f0,
    s0,
    n0,
):
    with dolfin.HDF5File(
        dolfin.MPI.comm_world, Path(filename).as_posix(), "w"
    ) as h5file:
        h5file.write(stimulation_cells, "/stimulation_cells")
        h5file.write(f0, "/f0")
        h5file.write(s0, "/s0")
        h5file.write(n0, "/n0")


def load_data(outdir="data"):
    fiber_file = Path(outdir) / "fibers.h5"
    if not fiber_file.is_file():
        raise FileNotFoundError(f"File {fiber_file} does not exist")

    geo = load_geometry(datadir=outdir)
    with dolfin.HDF5File(dolfin.MPI.comm_world, fiber_file.as_posix(), "r") as h5file:
        stimulation_cells = dolfin.MeshFunction("size_t", geo.heart_mesh, 3)
        h5file.read(stimulation_cells, "/stimulation_cells")

        W = dolfin.VectorFunctionSpace(geo.heart_mesh, "DG", 0)
        f0 = dolfin.Function(W)
        h5file.read(f0, "/f0")
        s0 = dolfin.Function(W)
        h5file.read(s0, "/s0")
        n0 = dolfin.Function(W)
        h5file.read(n0, "/n0")

    return Data(
        geo=geo,
        stimulation_cells=stimulation_cells,
        f0=f0,
        s0=s0,
        n0=n0,
    )


def generate_purkinje_network(mesh, outdir="data"):
    from fractal_tree import Mesh, FractalTreeParameters, generate_fractal_tree

    import meshio

    msh_file = Path(f"{outdir}/heart/biv_ellipsoid.msh")
    if not msh_file.is_file():
        raise FileNotFoundError(f"GMSH file {msh_file} does not exist")
    msh = meshio.read(msh_file)

    inds_lv = [
        i
        for i, x in enumerate(msh.cell_data["gmsh:physical"])
        if x[0] == msh.field_data["ENDO_LV"][0]
    ]
    connectivity_lv = np.vstack([msh.cells[i].data for i in inds_lv])

    inds_rv = [
        i
        for i, x in enumerate(msh.cell_data["gmsh:physical"])
        if x[0] == msh.field_data["ENDO_RV"][0]
    ]
    connectivity_rv = np.vstack([msh.cells[i].data for i in inds_rv])

    verts = msh.points

    init_node_lv = np.array([0, 1, 0])
    index_lv = np.linalg.norm(np.subtract(verts, init_node_lv), axis=1).argmin()

    init_node_rv = np.array([0, 1.5, 0.15])
    index_rv = np.linalg.norm(np.subtract(verts, init_node_rv), axis=1).argmin()

    mesh_lv = Mesh(
        verts=verts, connectivity=connectivity_lv, init_node=verts[index_lv, :]
    )
    param_lv = FractalTreeParameters(
        filename=f"{outdir}/biv-line-lv",
        N_it=15,
        initial_direction=np.array([1, 0, 0]),
        init_length=3.0,
        length=0.2,
    )

    # Next we create the Purkinje networks
    np.random.seed(123)
    lv_tree = generate_fractal_tree(mesh_lv, param_lv)

    mesh_rv = Mesh(
        verts=verts, connectivity=connectivity_rv, init_node=verts[index_rv, :]
    )
    param_rv = FractalTreeParameters(
        filename=f"{outdir}/biv-line-rv",
        N_it=20,
        initial_direction=np.array([1, 0, 0]),
        init_length=2.5,
        length=0.2,
    )

    # Next we create the Purkinje networks
    np.random.seed(123)
    rv_tree = generate_fractal_tree(mesh_rv, param_rv)

    V = dolfin.FunctionSpace(mesh, "DG", 0)
    f = dolfin.Function(V)
    from scipy.spatial import cKDTree

    dofs = V.tabulate_dof_coordinates()
    tree = cKDTree(dofs)

    stim_dofs = np.zeros(lv_tree.nodes.shape[0] + rv_tree.nodes.shape[0], dtype=int)
    for i, node in enumerate(lv_tree.nodes):
        stim_dofs[i] = tree.query(node)[1]

    for i, node in enumerate(rv_tree.nodes, start=lv_tree.nodes.shape[0]):
        stim_dofs[i] = tree.query(node)[1]

    f.vector()[stim_dofs] = 1.0

    stim = dolfin.MeshFunction("size_t", mesh, 3)
    stim.array()[:] = f.vector().get_local()
    return stim


def load_geometry(datadir="data") -> Geometry:
    # We need to create two meshes, one of BiV and one for
    # the surrounding torso
    datadir = Path(datadir)

    import cardiac_geometries as cg

    heart_path = datadir / "heart.h5"
    torso_path = datadir / "torso.h5"

    if not heart_path.is_file():
        heart = cg.create_biv_ellipsoid(
            outdir=heart_path.with_suffix(""), char_length=0.05
        )
        heart.save(heart_path)
    if not torso_path.is_file():
        torso = cg.create_biv_ellipsoid_torso(
            outdir=torso_path.with_suffix(""), char_length=0.6
        )
        torso.save(torso_path)

    heart = cg.geometry.Geometry.from_file(heart_path)
    torso = cg.geometry.Geometry.from_file(torso_path)

    return Geometry(
        heart_mesh=heart.mesh,
        heart_ffun=heart.ffun,
        heart_markers=heart.markers,
        torso_mesh=torso.mesh,
        torso_ffun=torso.ffun,
        torso_markers=torso.markers,
    )


def preprocess(outdir="data"):
    Path(outdir).mkdir(exist_ok=True)
    geo = load_geometry(datadir=outdir)

    markers = {
        "base": geo.heart_markers["BASE"][0],
        "rv": geo.heart_markers["ENDO_RV"][0],
        "lv": geo.heart_markers["ENDO_LV"][0],
        "epi": geo.heart_markers["EPI"][0],
    }

    stimulation_cells = generate_purkinje_network(geo.heart_mesh, outdir=outdir)

    f0, s0, n0 = ldrb.dolfin_ldrb(
        geo.heart_mesh,
        fiber_space="DG_0",
        ffun=geo.heart_ffun,
        markers=markers,
        alpha_endo_lv=60,
        alpha_epi_lv=-60,
    )
    for func, name in [(f0, "f0"), (s0, "s0"), (n0, "n0")]:
        with dolfin.XDMFFile(f"{outdir}/f0.xdmf") as f:
            f.write_checkpoint(func, name, 0.0, dolfin.XDMFFile.Encoding.HDF5, False)

    save_data(
        f"{outdir}/fibers.h5",
        stimulation_cells=stimulation_cells,
        f0=f0,
        s0=s0,
        n0=n0,
    )


def main():
    setup_general_parameters()
    # preprocess(refine=True)
    # data = load_data("data/coarse.h5")
    # data = load_data("data/refined.h5")

    # preprocess(outdir="data")
    data = load_data(outdir="data")

    data.geo.heart_mesh.coordinates()[:] /= 1000.0

    conductivities = setup_conductivities(data)
    cell_model = cbcbeat.Tentusscher_panfilov_2006_M_cell()
    time = dolfin.Constant(0.0)
    V = dolfin.FunctionSpace(data.geo.heart_mesh, "DG", 0)

    pulse = dolfin.CompiledExpression(
        dolfin.compile_cpp_code(cpp_stimulus).Stimulus(),
        element=V.ufl_element(),
        t=time._cpp_object,
        amplitude=30.0,
        duration=10.0,
        cell_data=data.stimulation_cells,
    )

    heart = cbcbeat.CardiacModel(
        data.geo.heart_mesh,
        time,
        conductivities.M_i,
        conductivities.M_e,
        cell_model,
        stimulus=pulse,
    )

    theta = 1.0
    params = cbcbeat.SplittingSolver.default_parameters()
    params["theta"] = theta
    params["CardiacODESolver"]["scheme"] = "GRL1"
    solver = cbcbeat.SplittingSolver(heart, params=params)

    (vs_, vs, vu) = solver.solution_fields()
    vs_.assign(heart.cell_models().initial_conditions())

    T = 1000.0
    k_n = 0.1

    solutions = solver.solve((0, T), k_n)

    u = vu.split()[1]

    VS0 = vs.function_space().sub(0)
    V = VS0.collapse()
    v = dolfin.Function(V)

    # Set-up object to optimize assignment from a function to subfunction
    assigner = dolfin.FunctionAssigner(V, VS0)
    assigner.assign(v, vs_.sub(0))

    for timestep, fields in solutions:
        # Store hdf5
        print("Solving on ", timestep)
        (t0, t1) = timestep
        assigner.assign(v, vs.sub(0))
        with dolfin.XDMFFile("results/results.xdmf") as res_file:
            res_file.write_checkpoint(v, "v", t0, dolfin.XDMFFile.Encoding.HDF5, True)
            res_file.write_checkpoint(u, "u", t0, dolfin.XDMFFile.Encoding.HDF5, True)


if __name__ == "__main__":
    # create_mesh()
    main()
