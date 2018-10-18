import networkx as nx

# ==============================================================================
# NODE DEFINITIONS
# ==============================================================================

# Added from Priestley-Taylor model
ET_c = "ET_c"
E_eq = "E_eq"
R_s = "R_s"
alpha = "alpha"
alpha_ms = "alpha_ms"
LAI = "LAI"
T_d = "T_d"
T_max = "T_max"
T_min = "T_min"

# Added from ASCE model
D_M = "D_M"
M = "M"
Y = "Y"
J = "J"
d_r = "d_r"
delta = "delta"
omega_s = "omega_s"
phi = "phi"
G_sc = "G_sc"
R_a = "R_a"
K_cb_min = "K_cb_min"
K_cb_max = "K_cb_max"
S_K_c = "S_K_c"
K_cb = "K_cb"
RH_min = "RH_min"
R_so = "R_so"
K_c_min = "K_c_min"
K_c_max = "K_c_max"
h = "h"
f_cd = "f_cd"
z = "z"
P = "P"
T = "T"
e_min = "e_min"
e_max = "e_max"
T_dew = "T_dew"
R_nl = "R_nl"
R_ns = "R_ns"
MEEVP = "MEEVP"
z_w = "z_w"
u_z = "u_z"
f_c = "f_c"
f_w = "f_w"
K_r = "K_r"
f_ew = "f_ew"
u_2 = "u_2"
C_d = "C_d"
C_n = "C_n"
R_n = "R_n"
e_a = "e_a"
e_s = "e_s"
Delta = "Delta"
gamma = "gamma"
G = "G"
ET_sz = "ET_sz"
K_e = "K_e"
# ==============================================================================


# ==============================================================================
# GRAPH HELPER FUNCTIONS
# ==============================================================================
def to_dotfile(nx_graph, filename):
    A = nx.nx_agraph.to_agraph(nx_graph)
    A.write(filename)


def to_png(nx_graph, filename):
    A = nx.nx_agraph.to_agraph(nx_graph)
    A.draw(filename, prog="dot")
# ==============================================================================


# ==============================================================================
# MODEL GRAPH GETTERS
# ==============================================================================
def priestley_taylor_graph():
    dg = nx.DiGraph()

    dg.add_nodes_from([
        ET_c, E_eq, R_s, alpha, alpha_ms, LAI, T_d, T_max, T_min
    ])

    dg.add_edges_from([
        (E_eq, ET_c), (T_max, ET_c), (R_s, E_eq), (alpha_ms, alpha),
        (alpha, E_eq), (LAI, alpha), (T_d, E_eq), (T_max, T_d), (T_min, T_d)
    ])

    return dg


def asce_graph():
    dg = nx.DiGraph()

    # Adding all nodes to ASCE digraph (nodes at end are from Priestley-Taylor)
    dg.add_nodes_from([
        D_M, M, Y, J, d_r, delta, omega_s, phi, G_sc, R_a, K_cb_min, K_cb_max,
        S_K_c, K_cb, RH_min, R_so, K_c_min, K_c_max, h, f_cd, z, P, T, e_min,
        e_max, T_dew, R_nl, R_ns, MEEVP, z_w, u_z, f_c, f_w, K_r, f_ew, u_2,
        C_d, C_n, R_n, e_a, e_s, Delta, gamma, G, ET_sz, K_e, ET_c, R_s, alpha,
        alpha_ms, LAI, T_max, T_min
    ])

    # Adding edges
    dg.add_edges_from([
        (D_M, J), (M, J), (Y, J), (J, d_r), (J, delta), (omega_s, R_a),
        (phi, R_a), (G_sc, R_a), (R_a, R_so), (K_cb_min, K_cb),
        (K_cb_max, K_cb), (R_s, f_cd), (R_s, R_ns), (alpha_ms, alpha),
        (LAI, alpha), (RH_min, K_c_max), (K_cb, K_c_max), (z, R_so), (z, P),
        (T_min, T), (T_min, e_min), (T_min, R_nl), (T_max, T),
        (T_max, e_max), (T_max, R_nl), (f_cd, R_nl), (alpha, R_ns),
        (h, K_c_max), (h, f_c), (K_c_max, f_c), (K_c_min, f_c),
        (f_w, f_ew), (f_c, f_ew), (u_z, u_2), (z_w, u_2),
        (MEEVP, K_c_max), (MEEVP, C_d), (MEEVP, C_n), (R_ns, R_n),
        (R_nl, R_n), (T_dew, e_a), (e_max, e_s), (e_min, e_s),
        (T, Delta), (P, gamma), (G, ET_sz), (gamma, ET_sz),
        (Delta, ET_sz), (e_s, ET_sz), (e_a, ET_sz), (e_a, R_nl),
        (R_n, ET_sz), (C_n, ET_sz), (C_d, ET_sz), (u_2, ET_sz),
        (f_ew, K_e), (K_r, K_e), (K_e, ET_c), (ET_sz, ET_c)
    ])

    return dg
# ==============================================================================


asce = asce_graph()
p_taylor = priestley_taylor_graph()
to_png(g, "asce-graph.png")
