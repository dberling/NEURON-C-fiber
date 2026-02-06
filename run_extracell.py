import os
import main
# turn on fixed time step simulation when testing
fixed_dt = os.environ.get("NEURON_FIXED_DT")
fixed_dt = 0.025
# position electrode in the 2/3 of parent 10um from C-fiber
extracell_rec=dict(electr_xyz_um=(80000,0,10), cond_SPERm=0.3)
if fixed_dt is not None:
    main.run(prot=0, dt=float(fixed_dt), extracell_rec=extracell_rec)
else:
    main.run(prot=0, extracell_rec=extracell_rec)
