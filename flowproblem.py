

import dolfin as df


class FlowProblem:
    """Base class for Stokes and Darcy simulators. Put physical quantities affecting both here."""
    # boundary conditions, specified as dolfin expressions
    # Flow boundary condition for velocity on domain boundary (avoid spaces for proper file path)
    # should be of the form u = (a_x + a_xy y, a_y + a_xy x)
    # do not forget the sign!!
    a_x = '+0.0'
    a_y = '+0.0'
    a_xy = '-2000.0'
    u_x = a_x + a_xy + '*x[1]'
    u_y = a_y + a_xy + '*x[0]'
    flowField = df.Expression((u_x, u_y), degree=2)
    # Pressure boundary condition field
    p_bc = a_x + '*x[0]' + a_y + '*x[1]' + a_xy + '*x[0]*x[1]'
    stressField = df.Expression(((p_bc, a_xy), (a_xy, p_bc)), degree=2)

    bodyForce = df.Constant((0.0, 0.0))  # right hand side; how is this treated in Darcy?

    class FlowBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            # SET FLOW BOUNDARIES HERE;
            # pressure boundaries are the complementary boundary in Stokes and need to be specified below for Darcy
            out = (x[1] > 1.0 - df.DOLFIN_EPS or (x[1] < df.DOLFIN_EPS) or
                   x[0] > 1.0 - df.DOLFIN_EPS or (x[0] < df.DOLFIN_EPS))

            if x[0] < df.DOLFIN_EPS and x[1] < df.DOLFIN_EPS:
                out = False
            return out
    flowBoundary = FlowBoundary()

    class PressureBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            # Set pressure boundaries here -- only for Darcy! For Stokes they are the complementary to flow boundaries
            # Therefore we can use 'on_boundary'
            return (x[0] < df.DOLFIN_EPS) and (x[1] < df.DOLFIN_EPS)
    pressureBoundary = PressureBoundary()
    # pressureBoundary = 'origin'

