import scipy.integrate


class Integrator:
    def integrate(self, fun, t_span, y0, t_eval) -> tuple:
        sol = scipy.integrate.solve_ivp(fun=fun, t_span=t_span, y0=y0, t_eval=t_eval)
        return sol.t, sol.y
