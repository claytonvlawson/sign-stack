# artifacts

`baseline_v1.pt` is downloaded by `./run.sh` from the v1.0.0 release. It is the
6-epoch checkpoint described in `configs/baseline.json`.

sha256: `906d7e3eec2656181eb7841d68354f69cb8ff3489734fa7e5630afebd512fd39`

It is not committed to git: GitHub rejects files over 100 MB and a release asset
is the right place for a weight file. The evaluation report records this hash,
so a run can be tied to the exact weights that produced it.
