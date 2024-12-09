import pytest
import os

cwd = os.getcwd()
rootdir = os.path.join(cwd[: cwd.rfind(os.path.sep + "probfem")], "probfem")
fig2_path = os.path.join(rootdir, "experiments", "reproduction", "rmfem", "fig2")
fig3_path = os.path.join(rootdir, "experiments", "reproduction", "rmfem", "fig3")

# some code at the start of each script to suppress matplotlib from showing figures
prefix = ""
prefix += "import matplotlib\n"
prefix += "import warnings\n"
prefix += 'matplotlib.use("agg")\n'
prefix += 'warnings.filterwarnings("ignore", message="Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.")\n'

# some code at the end of each script to suppress matplotlib from showing figures
suffix = ""
suffix += "import matplotlib.pyplot as plt\n"
suffix += "plt.close()\n"


@pytest.mark.rmfem
@pytest.mark.reproduction
@pytest.mark.runs
def test_rmfem_reproduction_fig2_runs(monkeypatch):
    monkeypatch.chdir(fig2_path)
    exec(prefix + open("fig2.py").read() + suffix)


@pytest.mark.rmfem
@pytest.mark.reproduction
@pytest.mark.runs
def test_rmfem_reproduction_fig3_runs(monkeypatch):
    monkeypatch.chdir(fig3_path)
    exec(prefix + open("fig3.py").read() + suffix)
