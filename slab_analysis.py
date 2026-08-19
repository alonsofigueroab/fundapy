import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import io

    return io, mo, pd


@app.cell
def _(mo):
    otm_risa = mo.ui.text_area(label="Pegar datos de RISA", rows=10, full_width=True)
    otm_risa
    return (otm_risa,)


@app.cell
def _(io, otm_risa, pd):
    df_otm = pd.read_csv(io.StringIO(otm_risa.value), sep='\t', skipinitialspace=True)
    df_otm['FSVx'] = df_otm['Ms-xx[mt-m]'] / df_otm['Mo-xx[mt-m]']
    df_otm['FSVz'] = df_otm['Ms-zz[mt-m]'] / df_otm['Mo-zz[mt-m]']
    df_otm
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
