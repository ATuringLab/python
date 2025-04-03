import h5py

weights_path = "generator_c.h5"

try:
    with h5py.File(weights_path, 'r') as f:
        print("Estructura del archivo HDF5:")
        print(list(f.keys()))
except Exception as e:
    print(f"Error al leer el archivo: {e}")