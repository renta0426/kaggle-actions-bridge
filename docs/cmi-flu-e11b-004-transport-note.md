# CMI-Flu E11b 004 dependency-transport repair

E11b-003 reached authenticated TabPFN model access successfully, then Kaggle `SaveKernel` rejected the generated script with HTTP 400 and the bounded safe message `The kernel source must be less than 1 megabytes in size.` The generated 003 runtime SHA-256 was `8df0a934a1eadcce52335f8087db3820649accb492d72aa8ffe66062d4a14449`.

The source-size failure was caused by embedding the frozen 771,167-byte `tabpfn-8.5.0-py3-none-any.whl` as base64 in the script. This is an operational dependency-transport failure, not a TabPFN scientific result and not a Kaggle compute failure.

E11b-004 preserves the science commit, exact E11b source blobs, B2.1 Task1.1 feature/target/CV contract, TabPFN package version, official Kaggle Model checkpoint, CPU parameters, and promotion gates. It changes only package transport:

- the generated runtime source must be `< 900000` bytes;
- no wheel bytes may be embedded in the source;
- the private research kernel enables Internet only to retrieve `tabpfn==8.5.0` from PyPI;
- the downloaded wheel must have filename `tabpfn-8.5.0-py3-none-any.whl`, size `771167`, and SHA-256 `4c076a019cfa5520e9c41405cecda845bdd09d909ede4a60c43839bbc83bf7a0` before installation;
- both download and install use `--no-deps`; download additionally uses `--only-binary=:all:`;
- the model checkpoint remains the attached official Kaggle Model `prior-labsai/tabpfn-3/pytorch/default/1`, with the same filename/byte/SHA checks;
- write transport remains one direct `KaggleApi.kernels_push()` call;
- failed target 003 and fresh target 004 must both be confirmed absent by exact read-only metadata before the write;
- automatic compute retries and Competition submissions remain disabled.

Archived CMI-Flu rules permit reasonably accessible external data/models/tools and do not impose a private-research Notebook Internet prohibition. The official TabPFN Kaggle Model instructions also use `pip install tabpfn` while attaching the model checkpoint as a Kaggle Model input. This operational change therefore does not redefine the tested scientific mechanism.
