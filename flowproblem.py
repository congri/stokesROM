

import dolfin as df
import fenics_adjoint as dfa


class FlowProblem:
    """Base class for Stokes and Darcy simulators. Put physical quantities affecting both here."""
    # boundary conditions, specified as dolfin expressions
    # Flow boundary condition for velocity on domain boundary (avoid spaces for proper file path)
    # should be of the form u = (a_x + a_xy y, a_y + a_xy x)
    u_x = '0.0-2.0*x[1]'
    u_y = '1.0-2.0*x[0]'
    flowField = dfa.Expression((u_x, u_y), degree=2)
    # Pressure boundary condition field
    p_bc = '0.0'
    pressureField = dfa.Expression(p_bc, degree=2)

    bodyForce = df.Constant((0.0, 0.0))  # right hand side; how is this treated in Darcy?

    class FlowBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            # SET FLOW BOUNDARIES HERE;
            # pressure boundaries are the complementary boundary in Stokes and need to be specified below for Darcy
            return x[1] > 1.0 - df.DOLFIN_EPS or (x[1] < df.DOLFIN_EPS and x[0] > df.DOLFIN_EPS) or \
                   x[0] > 1.0 - df.DOLFIN_EPS or (x[0] < df.DOLFIN_EPS and x[1] > df.DOLFIN_EPS)
    flowBoundary = FlowBoundary()

    class PressureBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            # Set pressure boundaries here -- only for Darcy, where no exclusions are present.
            # Therefore we can use 'on_boundary'
            return x[0] < df.DOLFIN_EPS
    pressureBoundary = PressureBoundary()
    # pressureBoundary = 'origin'