# Folder: scripts/vector_db_init.py
import faiss
import numpy as np

# Dummy array for placeholder
dim = 384
index = faiss.IndexFlatL2(dim)
index.add(np.random.rand(1000, dim).astype('float32'))
faiss.write_index(index, "vector.index")