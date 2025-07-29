# Fit Continuum

This repository contains a small tool for interactively normalizing spectra. The main script is `norm.py` and an example data file `4.txt` is provided.

## Requirements

The script depends on the packages listed in `requirements.txt`:

- numpy
- matplotlib
- scipy
- specutils

Install them with:

```bash
pip install -r requirements.txt
```

## Usage

Run the normalizer pointing to a text file with two columns (wavelength and flux):

```bash
python norm.py 4.txt
```

### Interactive controls

- **Left click** add continuum points
- **Right click** remove a previously selected point
- **Enter** fit the continuum using a cubic spline
- **n** normalize the spectrum by the fitted continuum
- **r** reset the selection and plots
- **w** save the normalized spectrum to `<input>.nspec`
- **q** exit the program

