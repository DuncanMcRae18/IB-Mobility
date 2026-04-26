
import dolfin

__all__ = ['setup_dolfin_parameters']


def update_dolfin_external_includes(headers):
    ffc_param = dolfin.parameters["form_compiler"]
    current = set(x for x in ffc_param["external_includes"].split(":") if x)
    current.update(headers)
    ffc_param["external_includes"] = ":".join(current)


def setup_dolfin_parameters():
    parameters = dolfin.parameters
    parameters["refinement_algorithm"] = "plaza_with_parent_facets"
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["cpp_optimize_flags"] = '-O3 -march=native'

    update_dolfin_external_includes(["boost/math/special_functions/lambert_w.hpp"])
