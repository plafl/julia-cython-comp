# Introduction

This repository contains benchmark code to compare Cython with Julia
implementing an small subset of
[LightFM](https://github.com/lyst/lightfm). 
You can find the details on my 
[blog](https://plopezadeva.com/julia-first-impressions.html).

Julia code doesn't have external dependencies. For the Python
benchmarks installing LightFM should be enough (it should pull Numpy
and Scipy on the process).


# Usage
Download and uncompress the test data here on the repo base
directory. The data can be downloaded from
[here](http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html).

Preprocessing must be done just once for both languages.


To perform the Python benchmarks:

```
python preprocessing.py
python test.py
```

To perform the Julia benchmarks:

```
julia preprocessing.jl
julia --math-mode=fast --check-bounds=no --threads=auto test.jl
```

