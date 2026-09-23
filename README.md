# SimKit Tutorials

Step-by-step tutorial notebooks for [SimKit](https://github.com/otmanon/simkit),
a simulation toolkit for computer animation.

These live in their own repository (separate from the main `simkit` package) so
that cloning `simkit` stays lightweight — the notebooks carry embedded plots and
a ~30 MB `media/` folder that most users never need.

## Viewing the tutorials

The notebooks are rendered as HTML on the SimKit documentation site under the
**Tutorials** section:

- https://otmanon.github.io/simkit/

You can also open any `.ipynb` here on GitHub, or clone this repo and run them
locally with Jupyter.

## Running them locally

```bash
git clone https://github.com/otmanon/simkit-tutorials
cd simkit-tutorials
pip install "simkit[all] @ git+https://github.com/otmanon/simkit" jupyter
jupyter lab
```

`utils.py` holds shared visualization helpers used across notebooks, and
`media/` holds the images / animations they reference.

## How the website stays in sync

The documentation site renders each notebook's **stored outputs** — it does not
re-execute them (the tutorials rely on `polyscope` / `libigl` and interactive
viewers that can't run in headless CI). So to update what the website shows:

1. Edit a notebook.
2. **Re-run it locally** so its output cells (plots, printed values) refresh.
3. Commit the notebook *with its outputs* (and any new files under `media/`).
4. Push — the SimKit `Docs` workflow checks this repo out and rebuilds the site.

See [DESIGN.md](DESIGN.md) for the original per-tutorial design notes.


## Rules

Please respect the rules for all tutorials. 


0. First most important rule. These notebooks are explanatory. Code in them should be explanatory and should not call obscure functions burried in utils/.  When building a mechancial or thermoelectric system, you should always create a function XXX_system, that returns its energy_func, gradient_func, and hessian_func later used by newton. This explanatory code, if it's important to the tutorial should go in the main notebook. 




1. If a notebook provides external media such as a plot, picture, gif or mp4, it saves a copy of it to media/ . Use simkit/filesystem for creating nd saving that piece of media.


2. All notebooks should use unified code/methods from simkit/ other libraries the repo depends on. Helper functions individual notebook should be minimized and attempted to be joined with pre-existing closely matching functions in simkit/ if possible. 


3. No code block or result block without explaining what it does.


4. No non-jupyter notebook files in this repo.


5. All notebooks are numbered with 3 digits 00_ then given the experiment name such as 00_experiment_name, to show the ordering in which they were created. Each notebook needs a unique number and unique name. Its corresponding files will be saved in notebooks/results/00_experiment_name/, via lib.filesystem.notebook_results_path

6. if a function is judged not important enough to go in simkit/ and not crucial to the tutorial, but is a helper function to the notebook, it should go in /utils.py

7. Minimize bloat as much as possible. Keep text concise. No more than one paragraph at a time unless absolutely necessary between code blocks/visual blocks.


8. Make lots of visual artifacts/temporary validation visual artifacts. If the experiment uses a mesh, visualize a screenshot of the mesh. if it uses a scalar field, visualize the scalar field.  Save it as an output.



9. All the changeable parameters should be at the top of each ntoebook.

10. Functions should always decompose sums into individual components line by line. Energy func should have elastic, pinned, etc as their own lines.
The final return statement should be the total, and that too should first be its own line summing everything. 