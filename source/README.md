# Browseable source mirror

These files are byte-for-byte copies of the Python implementation, dependency lock,
adaptation notes and protocol in the public reproducibility archive. The complete
cohort, split arrays, manifests and score evidence are distributed in the archive
as GitHub Release assets. Extract that archive to execute `verify.py` or `reproduce.py`;
this source mirror alone is not a complete runnable dataset distribution.

`experiments/ctd_sgformer/ego_neighbor_model.py` defines the SGC-MLP pairwise decoder;
`models.py` contains the shared encoders and SGFormer adaptation with its retained
upstream MIT notice; `protocol.py` defines split, masking and evaluation primitives.
`reproduce.py` orchestrates fresh reruns of the original training functions.
`protocol.md` documents the exact experimental settings.

No blanket license over upstream code or source-database records is granted.
