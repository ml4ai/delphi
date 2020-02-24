import time



Alt_PETPT_bounds = {
    'PETPT::@global::petpt::0::tmax::-1':[0.0, 40.0],
    'PETPT::@global::petpt::0::tmin::-1':[0.0, 40.0],
    'PETPT::@global::petpt::0::srad::-1': [0.0, 30.0],
    'PETPT::@global::petpt::0::msalb::-1': [0.0, 1.0],
    'PETPT::@global::petpt::0::xhlai::-1': [0.0, 20.0]
}

Orig_PETPT_bounds = {
    "PETPT::@global::petpt::0::msalb::-1": [0, 1],
    "PETPT::@global::petpt::0::srad::-1": [1, 20],
    "PETPT::@global::petpt::0::tmax::-1": [-30, 60],
    "PETPT::@global::petpt::0::tmin::-1": [-30, 60],
    "PETPT::@global::petpt::0::xhlai::-1": [0, 20],
}

PETASCE_bounds = {
    "PETASCE_simple::@global::petasce::0::doy::-1": [1, 365],
    "PETASCE_simple::@global::petasce::0::meevp::-1": [0, 1],
    "PETASCE_simple::@global::petasce::0::msalb::-1": [0, 1],
    "PETASCE_simple::@global::petasce::0::srad::-1": [1, 30],
    "PETASCE_simple::@global::petasce::0::tmax::-1": [-30, 60],
    "PETASCE_simple::@global::petasce::0::tmin::-1": [-30, 60],
    "PETASCE_simple::@global::petasce::0::xhlai::-1": [0, 20],
    "PETASCE_simple::@global::petasce::0::tdew::-1": [-30, 60],
    "PETASCE_simple::@global::petasce::0::windht::-1": [0.1, 10],  # HACK: has a hole in 0 < x < 1 for petasce__assign__wind2m_1
    "PETASCE_simple::@global::petasce::0::windrun::-1": [0, 900],
    "PETASCE_simple::@global::petasce::0::xlat::-1": [3, 12],     # HACK: south sudan lats
    "PETASCE_simple::@global::petasce::0::xelev::-1": [0, 6000],
    "PETASCE_simple::@global::petasce::0::canht::-1": [0.001, 3],
}

# TODO Here: lots of sensitivity analysis
