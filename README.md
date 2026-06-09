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
