def extract_marker_mesh_centroids(mesh):
    marker_meshes = mesh.split()
    centroids = [m.center_mass for m in marker_meshes]
    return centroids
