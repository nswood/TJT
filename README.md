# FlatSamples

Use remake_dist.py to make training data from the flat samples.
Edit pf.json to change what jet/particle information is used.
The output is an h5py file with each row corresponding to one jet. 

Each row will be in the form [jet info, pt (particles 0 through N), eta (particles 0 through N), ... , label (1 for signal, 0 for background)]

IN_FlatTau_v1p1.py is a basic interaction network that will train directly on the output of remake_dist.py.
