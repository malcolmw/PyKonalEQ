import h5py
import numpy as np
import pykonal


class TTInventory(object):

    def __init__(self, path, mode="r"):
        self._mode = mode
        self._path = path
        self._f5 = h5py.File(path, mode=mode)

        
    def __del__(self):
        self.f5.close()

        
    def __enter__(self):
        return (self)

    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__del__()

        
    @property
    def f5(self):
        return (self._f5)

    
    @property
    def mode(self):
        return (self._mode)
  

    @property
    def max_coords(self):
        return (self.min_coords + self.node_intervals * (self.npts - 1))
    
    
    @property
    def min_coords(self):
        return (self.f5["meta/min_coords"][:])


    @mode.setter
    def mode(self, value):
        self._mode = value
        self.f5.close()
        self._f5 = h5py.File(self.path, mode=value)
        
        
    @property
    def node_intervals(self):
        return (self.f5["meta/node_intervals"][:])
    
    @property
    def nodes(self):
        if not hasattr(self, "_nodes"):
            nodes = [
                np.linspace(
                    self.min_coords[idx],
                    self.max_coords[idx],
                    self.npts[idx],
                    pykonal.constants.DTYPE_REAL
                )
                for idx in range(3)
            ]
            nodes = np.meshgrid(*nodes, indexing="ij")
            nodes = np.stack(nodes)
            nodes = np.moveaxis(nodes, 0, -1)
            self._nodes = nodes
            
        return (self._nodes)
    
    
    @property
    def npts(self):
        return (self.f5["meta/npts"][:])

    @property
    def path(self):
        return (self._path)


    def add(self, field, key):

        if "meta" not in self.f5.keys():
            group = self.f5.create_group("meta")
            group.attrs["coord_sys"] = field.coord_sys
            group.attrs["field_type"] = field.field_type
            for attr in ("min_coords", "node_intervals", "npts"):
                group.create_dataset(attr, data=getattr(field, attr))
                
        else:
            group = self.f5["meta"]
            for attr in ("coord_sys", "field_type"):
                assert getattr(field, attr) == group.attrs[attr]
            for attr in ("min_coords", "node_intervals", "npts"):
                assert np.all(getattr(field, attr) == group[attr][:])
            
        group = self.f5.require_group("data")
        group.create_dataset(key, data=field.values)

        return (True)
    
    
    def close(self):
        self.f5.close()
        return (True)


    def read(self, key, min_coords=None, max_coords=None):
        _coord_sys  = self.f5["meta"].attrs["coord_sys"]
        _field_type = self.f5["meta"].attrs["field_type"]
        _min_coords = self.f5["meta/min_coords"][:]
        _node_intervals = self.f5["meta/node_intervals"][:]
        _npts = self.f5["meta/npts"][:]

        if min_coords is not None:
            min_coords = np.array(min_coords)

        if max_coords is not None:
            max_coords = np.array(max_coords)

        if min_coords is not None and max_coords is not None:
            if np.any(min_coords >= max_coords):
                raise(ValueError("All values of min_coords must satisfy min_coords < max_coords."))

        if min_coords is not None:
            idx_start = (min_coords - _min_coords) / _node_intervals
            idx_start = np.floor(idx_start)
            idx_start = idx_start.astype(np.int32)
            idx_start = np.clip(idx_start, 0, _npts - 1)

        else:
            idx_start = np.array([0, 0, 0])

        if max_coords is not None:
            idx_end = (max_coords - _min_coords) / _node_intervals
            idx_end = np.ceil(idx_end) + 1
            idx_end = idx_end.astype(np.int32)
            idx_end = np.clip(idx_end, idx_start + 1, _npts)

        else:
            idx_end = _npts

        if _field_type == "scalar":
            field = pykonal.fields.ScalarField3D(coord_sys=_coord_sys)
        elif _field_type == "vector":
            field = pykonal.fields.VectorField3D(coord_sys=_coord_sys)
        else:
            raise (ValueError(f"Unrecognized field type: {_field_type}"))

        field.min_coords = _min_coords  +  idx_start * _node_intervals
        field.node_intervals = _node_intervals
        field.npts = idx_end - idx_start
        idxs = tuple(slice(idx_start[idx], idx_end[idx]) for idx in range(3))
        field.values = self.f5[f"data/{key}"][idxs]

        return (field)
