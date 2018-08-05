"""
Different implementations of the NeRF algorithm.
"""

__author__ = "Mohammed AlQuraishi"
__copyright__ = "Copyright 2018, Harvard Medical School"
__license__ = "MIT"

def static_point_to_coordinate(pt, max_num_steps, bucket_boundaries=None, name=None):
    """ Takes points from dihedral_to_point and sequentially converts them into the coordinates of a 3D structure.

        This version of the function statically unrolls the reconstruction layer to max_num_steps. It uses a 
        hierarchical conditional scheme to sporadically check that the maximum number of steps has not been
        reached, and if it has, it exits early. The frequency of checking is determined by bucket_boundaries.

    Args:
        pt: [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]

    Returns:
            [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS] 
    """

    with tf.name_scope(name, 'point_to_coordinate', [pt]) as scope:
        pt = tf.convert_to_tensor(pt, name='pt')

        # pad to maximum length
        cur_num_dihedral_steps = tf.shape(pt)[0]
        batch_size = pt.get_shape().as_list()[1]
        pt = tf.pad(pt, [[0, (max_num_steps * NUM_DIHEDRALS) - cur_num_dihedral_steps], [0, 0], [0, 0]])
        pt.set_shape(tf.TensorShape([max_num_steps * NUM_DIHEDRALS, batch_size, NUM_DIMENSIONS]))

        # generate first three dummy coordinates
        id_mat = np.identity(NUM_DIHEDRALS, dtype='float32')
        coords = []
        for row in id_mat:
            coord = tf.tile(row[np.newaxis], [batch_size, 1])
            coord.set_shape([batch_size, NUM_DIMENSIONS]) # needed to prevent TF from choking
            coords.append(coord)
            # NUM_DIHEDRALS x [BATCH_SIZE, NUM_DIMENSIONS] 

        # loop over NUM_STEPS, sequentially generating the coordinates for the whole batch
        if bucket_boundaries is None: bucket_boundaries = []
        augmented_bucket_boundaries = [0] + bucket_boundaries + [max_num_steps]
        buckets = [[idx * NUM_DIHEDRALS for idx in augmented_bucket_boundaries[i:i+2]] for i in range(len(augmented_bucket_boundaries) - 1)]
        d = tf.unstack(pt, name='d') # (NUM_STEPS x NUM_DIHEDRALS) x [BATCH_SIZE, NUM_DIMENSIONS]
        for b_idx, e_idx in buckets:
            pass_through_coords = [tf.zeros([batch_size, NUM_DIMENSIONS])] * (e_idx - b_idx)
            def real_coords():
                coords_buffer = coords[-3:]
                for idx in range(b_idx, e_idx):
                    a, b, c = coords_buffer[-3:]                                                           # NUM_DIHEDRALS x [BATCH_SIZE, NUM_DIMENSIONS]
                    vec_bc = l2_normalize(c - b, 1, name='vec_bc')                                         # [BATCH_SIZE, NUM_DIMENSIONS]        
                    n = l2_normalize(tf.cross(b - a, vec_bc), 1, name='n')                                 # [BATCH_SIZE, NUM_DIMENSIONS]
                    m = tf.transpose(tf.stack([vec_bc, tf.cross(n, vec_bc), n]), perm=[1, 2, 0], name='m') # [BATCH_SIZE, NUM_DIMENSIONS, 3 TRANSFORMS]
                    coord = tf.add(tf.squeeze(tf.matmul(m, tf.expand_dims(d[idx], 2)), axis=2), c, name='coord') # [BATCH_SIZE, NUM_DIMENSIONS]
                    coords_buffer.append(coord)
                return coords_buffer[3:]

            new_coords = tf.cond(b_idx < cur_num_dihedral_steps, real_coords, lambda: pass_through_coords) # [BATCH_SIZE, NUM_DIMENSIONS]

            coords.extend(new_coords)
        # (NUM_STEPS x NUM_DIHEDRALS) x [BATCH_SIZE, NUM_DIMENSIONS] 

        # prune extraneous coords and reset back to actual length
        coords_pruned = tf.stack(coords[2:-1], name=scope)[:cur_num_dihedral_steps] # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS] 

        return coords_pruned

def dynamic_point_to_coordinate(pt, step_size=9, parallel_iterations=4, swap_memory=False, name=None):
    """ Takes points from dihedral_to_point and sequentially converts them into the coordinates of a 3D structure.
        
        This version of the function dynamically reconstructs the coordinates, using a fixed step_size of atomic
        coordinates to improve performance.

    Args:
        pt: [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]

    Returns:
            [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS] 
    """                             

    with tf.name_scope(name, 'point_to_coordinate', [pt]) as scope:
        pt = tf.convert_to_tensor(pt, name='pt')

        # initial three dummy coordinates
        Triplet = collections.namedtuple('Triplet', 'a, b, c')
        batch_size = pt.get_shape().as_list()[1]
        id_mat = np.identity(NUM_DIHEDRALS, dtype='float32')                                            
        init_coords = Triplet(*[tf.tile(row[np.newaxis], [batch_size, 1]) for row in id_mat])
                      # NUM_DIHEDRALS x [BATCH_SIZE, NUM_DIMENSIONS] 
        
        # extension function
        def extend(tri, pt):
            bc = l2_normalize(tri.c - tri.b, 1, name='bc')                                               # [BATCH_SIZE, NUM_DIMENSIONS]        
            n = l2_normalize(tf.cross(tri.b - tri.a, bc), 1, name='n')                                   # [BATCH_SIZE, NUM_DIMENSIONS]
            m = tf.transpose(tf.stack([bc, tf.cross(n, bc), n]), perm=[1, 2, 0], name='m')               # [BATCH_SIZE, NUM_DIMENSIONS, 3 TRANSFORMS]
            coord = tf.add(tf.squeeze(tf.matmul(m, tf.expand_dims(pt, 2)), axis=2), tri.c, name='coord') # [BATCH_SIZE, NUM_DIMENSIONS]
            return coord
        
        # loop over NUM_STEPS x NUM_DIHEDRALS, sequentially generating the coordinates for the whole batch; bunch of crud for dealing with 'step_size'
        i = tf.constant(1)
        s = tf.shape(pt)[0]
        r = ((step_size - (s % step_size)) % step_size) + 1
        pt = tf.pad(pt, [[0, r], [0, 0], [0, 0]])
        pt.set_shape(tf.TensorShape([None, batch_size, NUM_DIMENSIONS]))
        s_padded = tf.shape(pt)[0]
        coords_ta = tf.TensorArray(tf.float32, size=s_padded, tensor_array_name='coordinates_array')
        
        def body(i, tri, coords_ta): # (NUM_STEPS x NUM_DIHEDRALS) x [BATCH_SIZE, NUM_DIMENSIONS] 
            q = collections.deque([tri.a, tri.b, tri.c])
            for j in range(step_size):
                coord = extend(Triplet(*list(q)), pt[i - 1 + j])
                coords_ta = coords_ta.write(i + j, coord)
                q.popleft()
                q.append(coord)

            return [i + step_size, Triplet(*list(q)), coords_ta]

        limit = s_padded - step_size + 1
        _, _, coords = tf.while_loop(lambda i, _1, _2: i < limit, body, 
                                     [i, init_coords, coords_ta.write(0, init_coords.c)],
                                     parallel_iterations=parallel_iterations, swap_memory=swap_memory)
                       # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS] 

        coords_pruned = coords.stack()[:s]
        return tf.identity(coords_pruned, name=scope)

def simple_static_point_to_coordinate(pt, max_num_steps, name=None):
    """ Takes points from dihedral_to_point and sequentially converts them into the coordinates of a 3D structure.

        This version of the function statically unrolls the reconstruction layer to max_num_steps, and proceeds to
        reconstruct every residue without any conditionals for early exit. Note that it automatically pads and 
        unpads the input tensor to give back its original leng

    Args:
        pt: [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]

    Returns:
            [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS] 
    """

    with tf.name_scope(name, 'point_to_coordinate', [pt]) as scope:
        pt = tf.convert_to_tensor(pt, name='pt')

        # pad to maximum length
        cur_num_dihedral_steps = tf.shape(pt)[0]
        batch_size = pt.get_shape().as_list()[1]
        pt = tf.pad(pt, [[0, (max_num_steps * NUM_DIHEDRALS) - cur_num_dihedral_steps], [0, 0], [0, 0]])
        pt.set_shape(tf.TensorShape([max_num_steps * NUM_DIHEDRALS, batch_size, NUM_DIMENSIONS]))

        # generate first three dummy coordinates
        id_mat = np.identity(NUM_DIHEDRALS, dtype='float32')
        coords = []
        for row in id_mat:
            coord = tf.tile(row[np.newaxis], [batch_size, 1])
            coord.set_shape([batch_size, NUM_DIMENSIONS]) # needed to prevent TF from choking
            coords.append(coord)
            # NUM_DIHEDRALS x [BATCH_SIZE, NUM_DIMENSIONS] 
        
        # loop over NUM_STEPS, sequentially generating the coordinates for the whole batch
        for d in tf.unstack(pt, name='d'): # (NUM_STEPS x NUM_DIHEDRALS) x [BATCH_SIZE, NUM_DIMENSIONS]
            a, b, c = coords[-3:]                                                                   # NUM_DIHEDRALS x [BATCH_SIZE, NUM_DIMENSIONS]
            vec_bc = l2_normalize(c - b, 1, name='vec_bc')                                          # [BATCH_SIZE, NUM_DIMENSIONS]        
            n = l2_normalize(tf.cross(b - a, vec_bc), 1, name='n')                                  # [BATCH_SIZE, NUM_DIMENSIONS]
            m = tf.transpose(tf.stack([vec_bc, tf.cross(n, vec_bc), n]), perm=[1, 2, 0], name='m')  # [BATCH_SIZE, NUM_DIMENSIONS, 3 TRANSFORMS]
            coord = tf.add(tf.squeeze(tf.matmul(m, tf.expand_dims(d, 2)), axis=2), c, name='coord') # [BATCH_SIZE, NUM_DIMENSIONS]
            coords.append(coord)
            # (NUM_STEPS x NUM_DIHEDRALS) x [BATCH_SIZE, NUM_DIMENSIONS] 

        # prune extraneous coords and reset back to actual length
        coords_pruned = tf.stack(coords[2:-1], name=scope)[:cur_num_dihedral_steps] # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS] 

        return coords_pruned

def simple_dynamic_point_to_coordinate(pt, parallel_iterations=1, swap_memory=False, name=None):
    """ Takes points from dihedral_to_point and sequentially converts them into the coordinates of a 3D structure.
    
        This version of the function dynamically reconstructs the coordinatess one atomic coordinate at a time.
        It is less efficient than the version that uses a fixed number of atomic coordinates, but the code is 
        much cleaner here.

    Args:
        pt: [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]

    Returns:
            [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS] 
    """                             

    with tf.name_scope(name, 'point_to_coordinate', [pt]) as scope:
        pt = tf.convert_to_tensor(pt, name='pt')

        # initial three dummy coordinates
        Triplet = collections.namedtuple('Triplet', 'a, b, c')
        batch_size = pt.get_shape().as_list()[1]
        id_mat = np.identity(NUM_DIHEDRALS, dtype='float32')                                            
        init_coords = Triplet(*[tf.tile(row[np.newaxis], [batch_size, 1]) for row in id_mat])
                      # NUM_DIHEDRALS x [BATCH_SIZE, NUM_DIMENSIONS] 
        
        # extension function
        def extend(tri, pt):
            bc = l2_normalize(tri.c - tri.b, 1, name='bc')                                               # [BATCH_SIZE, NUM_DIMENSIONS]        
            n = l2_normalize(tf.cross(tri.b - tri.a, bc), 1, name='n')                                   # [BATCH_SIZE, NUM_DIMENSIONS]
            m = tf.transpose(tf.stack([bc, tf.cross(n, bc), n]), perm=[1, 2, 0], name='m')               # [BATCH_SIZE, NUM_DIMENSIONS, 3 TRANSFORMS]
            coord = tf.add(tf.squeeze(tf.matmul(m, tf.expand_dims(pt, 2)), axis=2), tri.c, name='coord') # [BATCH_SIZE, NUM_DIMENSIONS]
            return coord
        
        # loop over NUM_STEPS x NUM_DIHEDRALS, sequentially generating the coordinates for the whole batch
        i = tf.constant(1)
        s = tf.shape(pt)[0]
        coords_ta = tf.TensorArray(tf.float32, size=s, tensor_array_name='coordinates_array')
        
        def body(i, tri, coords_ta): # (NUM_STEPS x NUM_DIHEDRALS) x [BATCH_SIZE, NUM_DIMENSIONS] 
            coord = extend(tri, pt[i - 1])
            return [i + 1, Triplet(tri.b, tri.c, coord), coords_ta.write(i, coord)]
            
        _, _, coords = tf.while_loop(lambda i, _1, _2: i < s, body, 
                                     [i, init_coords, coords_ta.write(0, init_coords.c)],
                                     parallel_iterations=parallel_iterations, swap_memory=swap_memory)
                       # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS] 

        return coords.stack(name=scope)
