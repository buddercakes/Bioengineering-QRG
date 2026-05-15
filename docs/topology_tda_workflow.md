Python-First Biomedical TDA Workflow
====================================

This guide turns topological data analysis (TDA) into a practical workflow for
biomedical engineers, geneticists, microbiologists, and neurologists. It assumes
you want an analysis that can move from raw biomedical measurements to
persistent-homology features, visual diagnostics, and cohort-level comparisons.

When to use TDA
---------------

Use TDA when the biological question is more about shape, connectivity, cycles,
or branching than a single mean value.

| Data type | TDA question | Biological interpretation |
| --- | --- | --- |
| EMG or tremor traces | Do phase-space loops persist across scales? | Recurrent motor dynamics or pathological oscillations |
| EEG, ECoG, or fMRI windows | Does network topology change before events? | Seizure, sleep-stage, or cognitive-state transitions |
| Single-cell RNA-seq | Does the cell-state cloud branch or loop? | Differentiation, cycling, or trajectory ambiguity |
| Microbiome abundance profiles | Are communities clustered, bridged, or cyclic? | Enterotypes, dysbiosis gradients, ecological feedback |
| Protein conformations | Which cavities or loops remain stable? | Binding pockets, folding intermediates, steric constraints |

Core pipeline
-------------

1. Define the biological unit of analysis.
   Decide whether one point is a time window, cell, patient, microbial sample,
   protein conformation, or graph node.
2. Clean and normalize measurements.
   Remove artifacts, standardize features, and keep a record of clinical or
   experimental covariates.
3. Embed the data into a point cloud or graph.
   Examples include delay embeddings for time series, PCA/UMAP coordinates for
   single-cell data, or graph distances for connectomes.
4. Compute persistent homology.
   Track connected components, loops, and voids across distance thresholds.
5. Convert topology into interpretable features.
   Use persistence diagrams, barcodes, persistence images, landscapes, or Betti
   curves.
6. Validate biologically.
   Compare against labels, perturbations, null models, technical replicates, and
   known physiology.

Recommended Python stack
------------------------

| Task | Package |
| --- | --- |
| Arrays and tables | `numpy`, `pandas` |
| Scaling and machine learning | `scikit-learn` |
| Persistent homology | `ripser`, `gudhi`, `giotto-tda` |
| Diagram comparison | `persim` |
| Networks | `networkx` |
| Plotting | `matplotlib`, `seaborn` |

Minimal install command:

```bash
python -m pip install numpy pandas scikit-learn matplotlib seaborn ripser persim giotto-tda networkx
```

Workflow template
-----------------

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt

# 1. Load a samples-by-features matrix.
# Rows might be cells, patients, sliding time windows, or microbial samples.
X = pd.read_csv("biomedical_matrix.csv", index_col=0)
metadata = pd.read_csv("metadata.csv", index_col=0)

# 2. Normalize features so topology is not dominated by measurement scale.
X_scaled = StandardScaler().fit_transform(X)

# 3. Optional denoising or visualization embedding.
# Persistent homology can be run on the full matrix, but a conservative PCA
# projection can help inspect structure and reduce technical noise.
X_embed = PCA(n_components=10, random_state=0).fit_transform(X_scaled)

# 4. Compute persistence diagrams.
result = ripser(X_embed, maxdim=2)
diagrams = result["dgms"]

# 5. Visualize H0, H1, and H2 features.
plot_diagrams(diagrams, show=True)
plt.show()
```

Interpreting the output
-----------------------

| Diagram | Meaning | Biomedical reading |
| --- | --- | --- |
| H0 | Connected components | Clusters, subpopulations, patient strata |
| H1 | Loops | Rhythmic dynamics, feedback, recurrent trajectories |
| H2 | Voids | Cavities, excluded states, hollow conformational regions |
| Long lifetime | Feature persists across scales | More likely biological structure than noise |
| Short lifetime | Feature dies quickly | Often noise, local variation, or sampling artifact |

Time-series recipe: EMG, EEG, tremor, or physiology
---------------------------------------------------

For a single-channel signal, first reconstruct state space with a delay
embedding. Each point becomes a short memory of the signal.

```python
import numpy as np
from ripser import ripser
from persim import plot_diagrams


def delay_embedding(signal, dimension=3, delay=10):
    signal = np.asarray(signal, dtype=float)
    n_vectors = len(signal) - (dimension - 1) * delay
    if n_vectors <= 0:
        raise ValueError("Signal is too short for the requested embedding.")
    return np.column_stack([
        signal[i * delay : i * delay + n_vectors]
        for i in range(dimension)
    ])

# Example: one cleaned EMG or EEG channel.
signal = np.loadtxt("clean_signal.csv", delimiter=",")
point_cloud = delay_embedding(signal, dimension=4, delay=12)
diagrams = ripser(point_cloud, maxdim=1)["dgms"]
plot_diagrams(diagrams, show=True)
```

Practical interpretation:

- A strong H1 loop can indicate periodic or quasi-periodic dynamics.
- A changing H1 lifetime across sliding windows can indicate transition into or
  out of tremor, seizure, or other rhythmic pathology.
- Always compare against shuffled, phase-randomized, or amplitude-matched null
  signals before claiming biological meaning.

Single-cell or genomics recipe
------------------------------

For expression matrices, rows are cells or samples and columns are genes or
features. Apply standard quality control before TDA.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ripser import ripser

# expression: cells x genes after filtering and normalization
expression = pd.read_csv("expression_matrix.csv", index_col=0)
X = StandardScaler().fit_transform(expression)
X_pca = PCA(n_components=30, random_state=0).fit_transform(X)
diagrams = ripser(X_pca, maxdim=1)["dgms"]
```

Practical interpretation:

- H0 features can represent cell populations or sample strata.
- H1 loops may reflect cell-cycle structure, recurrent state transitions, or
  trajectory uncertainty.
- Mapper graphs are often useful when developmental trajectories branch rather
  than form simple clusters.

Microbiome recipe
-----------------

For compositional abundance data, avoid applying Euclidean geometry directly to
raw relative abundances. Use an appropriate transformation or distance first.

```python
import numpy as np
from sklearn.metrics import pairwise_distances
from ripser import ripser

abundance = pd.read_csv("microbiome_relative_abundance.csv", index_col=0)

# Simple centered-log-ratio style transform with a pseudocount.
pseudocount = 1e-6
log_abundance = np.log(abundance + pseudocount)
clr = log_abundance.sub(log_abundance.mean(axis=1), axis=0)

distance_matrix = pairwise_distances(clr, metric="euclidean")
diagrams = ripser(distance_matrix, distance_matrix=True, maxdim=1)["dgms"]
```

Practical interpretation:

- H0 features can separate community states or enterotype-like clusters.
- H1 loops can suggest gradients, cyclical ecological transitions, or sampling
  around excluded community states.
- Validate against sequencing depth, batch effects, diet, medication, and site.

Turning diagrams into machine-learning features
-----------------------------------------------

Persistent homology is often most useful when converted into stable numeric
features for prediction or statistical testing.

```python
from persim import PersistenceImager

# Fit on H1 diagrams from training data, then transform diagrams into images.
pimgr = PersistenceImager(pixel_size=0.05)
pimgr.fit(diagrams[1])
image = pimgr.transform(diagrams[1])

# Flatten for machine learning.
tda_features = image.ravel()
```

Common feature choices:

- Maximum H1 lifetime for rhythmic or recurrent dynamics.
- Number of long-lived H0 components for subpopulation structure.
- Total persistence as a summary of topological complexity.
- Persistence images or landscapes for classifier-ready vectors.
- Betti curves for scale-dependent topology.

Validation checklist
--------------------

- Compare disease versus control, pre versus post intervention, or time-windowed
  trajectories.
- Use permutation tests, bootstrap resampling, and null models.
- Check whether topology changes after artifact removal or batch correction.
- Report preprocessing, embedding dimension, delay, distance metric, `maxdim`,
  and filtration choice.
- Treat topology as evidence of structure, not as automatic proof of mechanism.

Biomedical caution notes
------------------------

- Topology is sensitive to preprocessing choices even though it is robust to
  small geometric perturbations.
- A persistent loop is not automatically a biological feedback loop; it may be a
  sampling, embedding, or batch artifact.
- Cohort imbalance, missingness, clinical confounding, and device artifacts can
  create convincing but non-biological topology.
- Interpret persistent features alongside domain knowledge, metadata, and
  conventional statistical models.

Minimal decision tree
---------------------

| If your data are... | Start with... | First topological target |
| --- | --- | --- |
| One-dimensional signals | Delay embedding | H1 loops over time |
| Multichannel signals | Sliding-window feature vectors | H0 clusters and H1 loops |
| Expression matrices | Scaled PCA space | H0 populations, H1 cycles |
| Microbiome profiles | CLR-transformed distances | H0 communities, H1 gradients |
| Networks | Graph distances or clique complexes | Connectedness and cycles |
| 3D structures | Coordinates or alpha complexes | Cavities, tunnels, voids |

Mental model
------------

The Python workflow is simple: represent each biological observation as a point,
choose a scientifically defensible distance, compute which connected components
and loops survive across scales, then test whether those features track biology
better than noise, artifacts, or batch effects.
