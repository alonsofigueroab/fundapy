import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import forallpeople as si
    from math import sqrt

    return si, sqrt


@app.cell
def _(si):
    si.environment('default')
    si.mm   = 0.001* si.m
    si.MPa  = 1e6 *si.Pa
    return


@app.cell
def _(si):
    # Parametros iniciales

    ## Hormigón

    fc       = 30    *si.MPa
    ɸsl      = 0.65
    λ_a      = 1
    ψ_a      = 1
    ψ_brg_sl = 1
    ψ_ed_V   = 1
    ψ_c_V    = 1.2
    ψ_h_V    = 1

    h_gr = 40    *si.mm

    ## Placa de anclaje
    w_bp = 350 *si.mm
    t_bp = 12  *si.mm

    ## Llave de corte

    h_sl = 150 *si.mm
    h_ef_sl = h_sl - h_gr
    b_sl = 0.5 *h_ef_sl
    t_sl = 12  *si.mm

    l_e  = h_ef_sl
    ca1  = 0.5 * (w_bp - t_sl)

    A_ef_sl  = 2* (2*t_sl * h_ef_sl) + 2* (t_sl * (b_sl - 2* t_sl) )

    Avc  = (1.5 * ca1 + b_sl + 1.5 * ca1) * (1.5 * ca1 + h_ef_sl) - A_ef_sl
    Avco = 4.5 * ca1**2
    return (
        A_ef_sl,
        Avc,
        Avco,
        ca1,
        fc,
        ɸsl,
        λ_a,
        ψ_a,
        ψ_brg_sl,
        ψ_c_V,
        ψ_ed_V,
        ψ_h_V,
    )


@app.cell
def _(A_ef_sl, Avc, Avco, V_b, fc, ɸsl, ψ_a, ψ_brg_sl, ψ_c_V, ψ_ed_V, ψ_h_V):
    V_brg_sl = 1.7 * fc * A_ef_sl * ψ_brg_sl * ɸsl
    V_cb_sl  = Avc/Avco * ψ_a * ψ_ed_V * ψ_c_V * ψ_h_V * V_b * ɸsl
    return (V_brg_sl,)


@app.cell
def _(ca1, fc, sqrt, λ_a):
    # V_b_1 = 0.6 * (l_e/d_a)**0.2 * sqrt(d_a) * λ_a * sqrt(fc) * ca1**1.5
    V_b = 3.7 * λ_a * sqrt(fc) * ca1**1.5
    return (V_b,)


@app.cell
def _(V_brg_sl):
    V_brg_sl
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
