import os
import main
# turn on fixed time step simulation when testing
fixed_dt = 0.025
if fixed_dt is not None:
    main.run(prot=0, dt=float(fixed_dt))
else:
    main.run(prot=0)
