
# Setup Quarto to run these examples locally

To run the examples locally, you need [Quarto](https://quarto.org) installed on your system,
see [the Quarto Julia installation guide](https://quarto.org/docs/computations/julia.html#installation).

Then, you have instantiate the environment in the example folder:
Run `julia --project=@. -e"using Pkg; Pkg.instantiate();"` in the `examples/` folder on terminal.

After that, running e.g. `quarto render examples/Rosenbrock.qmd` renders a new resulting
`.md` file into the `docs/src/examples/` folder.

You can also render all examples by running `docs/make.jl --quarto` from terminal.
This also renders the HTML pages in `docs/build` and might be nicer to read/look at the examples.
Calling the same command later only renders those examples whose `.qmd` file was changed in the meantime.
