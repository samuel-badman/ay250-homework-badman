import numpy as np
from numpy.linalg import norm,inv
from math import pi,sin,cos,tan
from multiprocessing import Pool,cpu_count
from functools import partial
from sys import stdout

#_______________________________________________________________________________
def get_shape() :
    """
    Return an array of sets of 3 vertices, each describing a triangular finite 
    plane which forms part of the surface of a solid body.
    
    The solid body here is a 3x1x1 satellite bus with two 3x3 solar panels 
    parallel to the 1x1 face of the satellite bus.  
    
    Parameters
    ----------
    
    Keyword Arguments
    -----------------
    
    Returns
    -------
    tripoint_arr : list of [3x3] numpy arrays 
        [U] - 1U = 10cm, each [3x3] is vertices of a triangle
        
    Warnings
    --------
    
    Example
    -------
    >>> import ray_tracer as rt
    >>> tri_arr = rt.get_shape()
    >>> print(tri_arr[0])
    array([[ 0.5, -0.5,  1.5],
           [ 0.5,  0.5,  1.5],
           [-0.5, -0.5,  1.5]])
      
    References
    ----------
    
    Notes
    -----
    07/05/18 - Sam Badman - first commit
    """

    
    # Vertices of solid body:
    corners=np.array([[ 0.5, 0.5, 1.5],[ 0.5,-0.5, 1.5],[-0.5,-0.5, 1.5],
                      [-0.5, 0.5, 1.5],[ 0.5, 0.5,-1.5],[ 0.5,-0.5,-1.5],
                      [-0.5,-0.5,-1.5],[-0.5, 0.5,-1.5],[ 1.5, 0.5, 1.5],
                      [ 1.5, 3.5, 1.5],[-1.5, 3.5, 1.5],[-1.5, 0.5, 1.5],
                      [ 1.5,-0.5, 1.5],[ 1.5,-3.5, 1.5],[-1.5,-3.5, 1.5],
                      [-1.5,-0.5, 1.5]])

    # Each triangular plane is represented by 3 points. Break each rectangular face into 
    # two triangles. Make a list of indices of these trios of points
    triface_indices = [[1,0,2],[3,0,2],[0,1,4],[5,1,4],[4,7,0],[3,7,0],
                       [3,7,2],[6,2,7],[6,5,2],[1,2,5],[5,6,4],[7,4,6],
                       [9,10,8],[11,10,8],[15,12,14],[13,14,12]]

    # make array of corner values for each triangle
    tripoint_arr = []
    for el in triface_indices :
        tripoints = np.array([corners[ind] for ind in el])
        tripoint_arr.append(tripoints)
        
    # Return a list of body face triangle coordinates
    return tripoint_arr

#_______________________________________________________________________________
def rot_3d_aa(axis, angle, active = True) :
    """Return the rotation matrix specified by a rotation axis and an angle

    Following the Euler Axis-Angle parameterisation of rotations, this function  
    takes a given numpy 3x1 array (unit vector) specifying the axis to be
    rotated about, and an angle given in degrees, and returns the resulting
    rotation matrix. 
    
    Active Convention (Default) : The returned rotation matrix is the one which
    rotates a given vector in a fixed coordinate system about the supplied axis 
    expressed in that fixed coordinate system by a the supplied angle
    
    Passive Convention (active = False) : The returned rotation matrix is the 
    the matrix which converts a vector from one coordinate system to another, 
    where the intial coordinate axis are rotated by the axis and angle given 
    as expressed in the intial coordinate frame. This matrix is simply the
    transpose of the active convention.
     
    Parameters
    ----------
    axis : [3x1] float
        [n/a] unit vector specifying axis of rotation. 
    angle : float
        [deg] angle of rotation

    Keywords Arguments
    ------------------
    active : boolean
        [n/a] True (Default) - produces active rotation matrix, False - produces
              passive rotation matrix.
    
    Returns
    -------
    rot : 3x3 float
        [n/a] rotation matrix corresponding to input axis and angle.

    Warnings
    --------
    Input: axis and Output: rot are numpy arrays

    Example
    -------
    Return the rotation matrix for 90 deg about the z-axis
    
    >>> import curie_toolbox as ct
    >>> import numpy as np
    >>> ax = np.array([ 0., 0., 1.])
    >>> ct.rot_3d_aa(ax, 90.0)
    array([[ 0. -1.  0.]
           [ 1.  0.  0.]
           [ 0.  0.  1.]])

    References
    ----------

    Notes
    -----
    07/11/17, SB, Initial Commit 
    """
    theta = pi/180.0*angle             #convert to radians
    k  = axis/norm(axis) #ensure normalised
    
    rot =  np.array([[cos(theta) + k[0]**2*(1.0-cos(theta)) , k[0]*k[1]* 
                      (1.0 - cos(theta)) - k[2]*sin(theta) ,  k[0]*k[2]*
                      (1.0 - cos(theta)) + k[1]*sin(theta)               ], 
                     [ k[0]*k[1]* (1.0 - cos(theta)) + k[2]*sin(theta) , 
                       cos(theta) + k[1]**2*(1.0 - cos(theta)) , 
                       k[1]*k[2]*(1.0 - cos(theta)) - k[0]*sin(theta)    ], 
                     [ k[0]*k[2]*(1.0 - cos(theta)) - k[1]*sin(theta) , 
                       k[1]*k[2]*(1.0 - cos(theta)) + k[0]*sin(theta) , 
                       cos(theta) + k[2]**2*(1.0 - cos(theta))           ]])
    
    rot[abs(rot) < 1e-16] = 0. #Get rid of floating point errors.
    if active == True :
        return rot
    elif active == False :
        return np.transpose(rot)
    else : raise ValueError('\'active\' should be a boolean')

#_______________________________________________________________________________

def ray_coords(az, el, yp, zp, Ny, Nz, rc) :
    '''Generate ray plane - source of rays simulating impinging atmosphere
    
    Generate a finite plane with center coordinate initially at [rc,0,0],
    parallel to yz plane with extend [yp,zp]. The plane is populated with
    Ny points in the y direction and Nz points in the z direction. A mirror
    image plane with center coordinate [-rc,0,0] is also generated. The planes
    are then rotated into place by an elevation (rotation about the -y axis) of
    angle el (degrees) followed by an azimuthal rotation (about the +z axis) of
    angle az (degrees).
    
    Arguments:
    ----------
    az : float
        [deg] : azimuth angle of ray plane
        
    el : float
        [deg] : elevation angle of ray plane
        
    yp : float
        [U=10cm] : extent of ray plane in y direction (before rotation)
    
    zp : float 
        [U=10cm] : extent of ray plane in z direction (before rotation)
        
    Ny : int
        [] : number of rays in ray plane in y direction
        
    Nz : int
        [] : number of rays in ray plane in z direction
    
    rc : float
        [U=10cm] : distance of central point of plane to origin of solid body
    

    Keywords:
    ---------
    
    Returns
    -------
    rp - [(Ny+1)*(Nz+1),3]
        [U=10cm] : ndarray of vectors of incident rays.

    rp - [(Ny+1)*(Nz+1),3]
        [U=10cm] : ndarray, incident plane reflected through the origin        - 
    
    Warnings
    --------
    For accurate results
        - rc should exceed the largest norm vertex on the solid
        - yp,zp should be larger than the largest vertex-vertex distance on
          the solid.
          
    Example
    -------
    >>> import ray_tracer as rt
    >>> rp,rp_refl = rt.ray_coords(0.,0.,3.0,3.0,20,20,3.0)
    >>> print(rp[0],rp_refl[0])
    [ 3.  -1.5 -1.5] [-3.  -1.5 -1.5]

    References
    ----------
    
    Notes
    -----
    07/05/18 - Sam Badman, first commit
    '''
    # body coordinate unit vectors
    (xhat,yhat,zhat) = (np.array([1.,0.,0.]), np.array([0.,1.,0.]),
                        np.array([0.,0.,1.])
                       ) 



    # Generate rotation matrix         
    # First rotate about y by elevation angle * -1, 
    #then rotate about z with the azimuth angle
    rot = np.dot(rot_3d_aa([0.,0.,1.],az),rot_3d_aa([0,1,0],-1.*el))      
            
    # Lists to place ray coordinates in
    rp = []
    rp_refl = []

    # Loop over plane, populate lists
    for ii in range(Ny+1) :
        ny = ii - Ny/2
        for jj in range(Nz+1) :
            nz = jj - Nz/2
            rp.append(rc*xhat + ny*yp/Ny * yhat + nz*zp/Nz * zhat)
            rp_refl.append(-rc*xhat + ny*yp/Ny * yhat + nz*zp/Nz * zhat) 
    rp = np.array(rp)
    rp_refl = np.array(rp_refl)

    # Rotate plane into correct orientation:
    rp = np.dot(rot,rp.T).T
    rp_refl = np.dot(rot,rp_refl.T).T
    
    # return these coordinates
    return rp, rp_refl

#_______________________________________________________________________________
def compute_intersections(ray_ind,az=30.,el=45.,yp=3.0,
                          zp=3.0,Ny=20,Nz=20,rc=3.0) :
    '''Trace ray, determine the plane in the body it intersects first 

    A given ray is specified from the ray coords from ray_tracer.ray_coords
    This specifies a finite line which is tested against each triangular plane
    in the body for intersection. If the list of intersecting planes is not 
    empty, the plane which intersects closest to the ray plane is determined
    and this plane is identified and returned, along with the coordinate of 
    intersection, and the coordinate on the ray plane where the ray originated.
    
    Arguments
    ---------
    ray_ind : int
        [] : the number ray to choose. Must be less than (Ny+1)*(Nz+1)
        
    Keywords
    --------
    az : float (Default : 30)
        [deg] : azimuth angle of ray plane
        
    el : float (Default : 45)
        [deg] : elevation angle of ray plane
        
    yp : float (Default : 3.0)
        [U=10cm] : extent of ray plane in y direction (before rotation)
    
    zp : float (Default : 3.0)
        [U=10cm] : extent of ray plane in z direction (before rotation)
        
    Ny : int (Default : 20)
        [] : number of rays in ray plane in y direction
        
    Nz : int (Default : 20)
        [] : number of rays in ray plane in z direction
    
    rc : float (Default : 3.0)
        [U=10cm] : distance of central point of plane to origin of solid body
        
    Returns
    -------
    face_number : int 
        [] : finite plane number from tripoint_arr. Empty string if no intersec
        
    r_int : [3x1] float
        [U,U,U] : coordinate where ray and plane intersect
        
    ray_point : [3x1] float
        [U,U,U] : coordinate where ray originated
        
    Warnings
    --------
    Recommend setting 
    >>> rc=1.*max(numpy.linalg.norm(rt.get_shape(),axis=1).flatten())
    and 
    yp,zp = 1.5*rc,1.5*rc
    to ensure plane envelops the solid body,
    
    Example
    -------
    >>> import ray_tracer as rt
    >>> rt.compute_intersections(0)
    # No intersections
    ('',
    array([-0.16855865, -1.82936819, -3.18198052]),
    array([3.50567596, 0.29195215, 1.06066017]))
    >>> rt.compute_intersections(200)
    # Intersects face 13 at coordinate [1.19032637, 0.51403017, 1.5       ]
    (13,
    array([1.19032637, 0.51403017, 1.5       ]),
    array([1.82026144, 0.87772335, 2.22738636]))

    References
    ----------
    Austin Nicholas,David Miller - Attitude and Formation Control Design and System 
    Simulation for a Three-Satellite CubeSat Mission
    http://ssl.mit.edu/files/website/theses/SM-2013-NicholasAustin.pdf
    
    Notes
    -----
    05/07/18 - Sam Badman - First commit
    '''
    # Get ray coords (start and end)
    rp,rp_refl = ray_coords(az=az,el=el,yp=yp,zp=zp,Ny=Ny,Nz=Nz,rc = rc)

    # Generate triangle finite planes
    tripoint_arr = get_shape()
    
    face_inds,ts = [],[]
    # Loop over all planes in solid body, and check for intersection with ray.
    for face_ind in range(len(tripoint_arr)) :
        ra,rb = rp[ray_ind],rp_refl[ray_ind]
        (r0,r1,r2) = (tripoint_arr[face_ind][0],tripoint_arr[face_ind][1],
                      tripoint_arr[face_ind][2])
        #Check if ray is parallel to face (avoid singular matrix)
        # normal vector to face
        n = np.cross(r1-r0,r2-r0)
        # dot product of n with ray. If 0, the ray and plane are parallel
        if np.dot(n,rb-ra) == 0 : continue
        else :
            # See reference for mathematics. [M^-1][V] = (t,u,v)
            # if t,u,v obey a set of constraints, ray and plane intersect
            matrix = np.array([[ra[0]-rb[0],r1[0]-r0[0],r2[0]-r0[0]],
                               [ra[1]-rb[1],r1[1]-r0[1],r2[1]-r0[1]],
                               [ra[2]-rb[2],r1[2]-r0[2],r2[2]-r0[2]]])
            vec  = np.array([[ra[0]-r0[0]],
                             [ra[1]-r0[1]],
                             [ra[2]-r0[2]]])
            mat_inv = inv(matrix)        
            vec_out = list(np.dot(mat_inv,vec))
            t,u,v = vec_out[0][0],vec_out[1][0],vec_out[2][0]
            if t < 1 and t > 0 and u < 1 and u > 0 and v < 1 and v > 0 and u+v <=1 :
                face_inds.append(face_ind)
                ts.append(t)

    # If no intersection, return empty string for face ID, and ray goes from
    # source plane to reflected plane
    if not ts : return '',rp_refl[ray_ind],rp[ray_ind]
    else :
    #True intersection is where t is the smallest
        face_number = face_inds[ts.index(min(ts))]
        t_val = ts[ts.index(min(ts))]
        r_int = ra + (rb-ra)*t_val
        return face_number,r_int,rp[ray_ind]
#_______________________________________________________________________________
def compute_drag(az=30.,el=45.,Ny=20,Nz=20,yp=3.0,zp=3.0,rc=3.0,viz=False) :
    ''' Iterate over rays, and process to compute drag force and drag torque
    
    After finding the intersection plane and intersection point for each ray,
    the drag force may be found by summing the incoming ray vector over all
    plane intersections. The total drag torque may be found by taking the 
    cross product of the intersection point with each drag force element and
    summing. Ray tracing is performed in parallel with multiprocessing.Pool.
    
    Arguments
    ---------
    
    Keywords
    --------
    az : float (Default : 30)
        [deg] : azimuth angle of ray plane
        
    el : float (Default : 45)
        [deg] : elevation angle of ray plane
        
    yp : float (Default : 3.0)
        [U=10cm] : extent of ray plane in y direction (before rotation)
    
    zp : float (Default : 3.0)
        [U=10cm] : extent of ray plane in z direction (before rotation)
        
    Ny : int (Default : 20)
        [] : number of rays in ray plane in y direction
        
    Nz : int (Default : 20)
        [] : number of rays in ray plane in z direction
    
    rc : float (Default : 3.0)
        [U=10cm] : distance of central point of plane to origin of solid body
    
    viz : Bool (Default : False)
        [] : switch to return more objects if required for plotting
        
    Returns
    -------
    force : [3x1] float
        [N,N,N] : Total drag force for inputted flow direction
    
    torque : [3x1] float
        [Nm,Nm,Nm] : Total drag torque for inputted flow direction
    
    hits : [n_intersections x 3] float (returned if viz == True)
        [U=10cm] : coordinates of intersections
    
    sources : [n_intersections x 3] float (returned if viz == True)
        [U=10cm] : initial coords of intersecting rays
    
    faces : [n_intersections] int (returned if viz == True)
        [] : list of faces which were intersected by each ray
    
    Warnings
    --------
    
    Example
    -------
    >>> import ray_tracer as rt
    >>> rt.compute_drag()
    (array([-179.42512366, -103.59114344, -207.18228689]),
    array([ 17.42638453, -94.03512574,  31.92587117]))
    >>> rt.compute_drag(az=0.0,el=90.0)
    (array([   0.,    0., -315.]), array([0., 0., 0.]))

    References
    ----------
    Austin Nicholas,David Miller - Attitude and Formation Control Design and System 
    Simulation for a Three-Satellite CubeSat Mission
    http://ssl.mit.edu/files/website/theses/SM-2013-NicholasAustin.pdf
    
    Notes
    -----
    05/07/18 - Sam Badman - First commit
    '''
    
    # Ray indices to loop over
    rays = range((Ny+1)*(Nz+1))
    
    # Lists to store for outputting
    hits = []
    sources = []
    faces = []
    
    # Drag force per array
    drag_mult = 1.0
    
    # Get ray origin coordinates
    rp,rp_refl = ray_coords(az=az,el=el,yp=yp,zp=zp,Ny=Ny,Nz=Nz,rc=rc)
    
    # Get unit vector of ray propagation direction
    ray_norm = rp_refl[0]-rp[0]
    ray_norm = ray_norm/norm(ray_norm)
    
    # Initialize Pool to compute ray intersections in parallel
    ray_tracer = Pool(cpu_count())
    results = ray_tracer.map(partial(compute_intersections,az=az,el=el,Ny=Ny,
                                     yp=yp,zp=zp,Nz=Nz,rc=rc),rays) 
    ray_tracer.close()
    
    # Pull out rays which did intersect
    for result in results :
        if result[0] != '' : 
            faces.append(result[0])
            hits.append(result[1])
            sources.append(result[2])        
    hits = np.array(hits)
    sources = np.array(sources)
    
    # Sum individual force elements to produce total drag force vector
    force = drag_mult*ray_norm*hits.shape[0]
    
    # Take cross product of each force element with intersection location
    # to get torque elements. 
    diff_torques = np.cross(hits,drag_mult*ray_norm)
    
    # Sum torque elements to get total torque
    torque = np.sum(diff_torques,axis=0)
    torque[abs(torque) < 1e-12] = 0.0
    
    if viz == True : return force,torque,hits,sources,faces
    else : return force, torque
#_______________________________________________________________________________
def tabulate_drag(Naz,Nel,Ny=20,Nz=20,yp=15.,zp=15.,rc=9.) :
    ''' Compute drag force and torque for flow dirs from around the unit sphere.
    
    Iterate over different azimuth and elevation angles, and for each flow
    direction, calculate the drag force and drag torque. Return the results
    tabulated in a 2x2 array,with a corresponding array containing the az,el
    angles.
    
    Arguments
    ---------
    Naz : int
        [] : number of azimuth angles to compute from [0:360] deg
        
    Nel : int
        [] : number of elevation angles to compute from [-90:90] deg
    
    Keywords
    --------
    Ny : int (Default : 20)
        [] : number of rays in ray plane in y direction
        
    Nz : int (Default : 20)
        [] : number of rays in ray plane in z direction
    
    yp : float (Default : 15.0)
        [U=10cm] : extent of ray plane in y direction (before rotation)
    
    zp : float (Default : 15.0)
        [U=10cm] : extent of ray plane in z direction (before rotation)
        
    rc : float (Default : 9.0)
        [U=10cm] : distance of central point of plane to origin of solid body
    
    Returns
    -------
    drag_table : [Naz x Nel] nested list, each list el is a tuple
        ([N,N,N],[Nm,Nm,Nm]) : table of tuples of force and torque
        
    angles_table : [Naz x Nel] nested list, each list el is a tuple
        (deg,deg) : each tuple contains azimuth and elevation angle
        
    Warnings
    --------
    
    Examples
    --------
    
    References
    ----------
    
    Notes
    -----
    05/07/18 - Sam Badman - First commit
    '''
    # Initialize output lists
    drag_table = []
    angles_table = []
    print("Calculating drag force and torque for :")
    # Loop over azimuth angles from 0-360 deg
    for az in np.arange(0,360+360/Naz,360/Naz) :
        temp = []
        temp2 = []
        # Loop over elevation angles from -90 to +90 deg
        for el in np.arange(-90,90+180/Nel,180/Nel) :
            stdout.write("\rAzimuth :"+str(az)+" deg, Elevation :"
                         +str(el)+" deg")
            # Compute force and torque for the given flow orientation
            f,t = compute_drag(az=az,el=el,Ny=Ny,Nz=Nz,yp=yp,zp=zp,rc=rc)
            temp.append((f,t))
            temp2.append((az,el))
        drag_table.append(temp)
        angles_table.append(temp2)
        
    return drag_table,angles_table
         


