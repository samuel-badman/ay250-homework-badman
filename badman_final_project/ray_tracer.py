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
    >>> rt.get_shape()
    [array([[ 0.5, -0.5,  1.5],
            [ 0.5,  0.5,  1.5],
            [-0.5, -0.5,  1.5]]),
            ...
            ...
            ...
            [ 1.5, -0.5,  1.5]])]
            
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
        [deg] : 
    

    Keywords:
    rc = 2.0 
        - distance of center of ray plane from origin of solid body 
        - for accurate results should be further away than futhest coordinate of
        solid body
    yp = 3.0, zp = 3.0
        - For az = el = 0.0 the plane is orientated in the yz plane. yp,zp gives
         the size of the finite plane in the yz directions before being rotated
        - this should be larger than the largest dimension of the solid object, 
        i.e a sphere of this diameter with its origin at the solid body origin 
        should completely envelop the solid body.
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

    # Get ray coords (start and end)
    rp,rp_refl = ray_coords(az=az,el=el,yp=yp,zp=zp,Ny=Ny,Nz=Nz,rc = rc)

    # Generate triangle finite planes
    tripoint_arr = get_shape()
    
    face_inds,ts = [],[]
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

    #True intersection is where t is the smallest
    if not ts : return '',rp,rp
    else :
        face_number = face_inds[ts.index(min(ts))]
        t_val = ts[ts.index(min(ts))]
        r_int = ra + (rb-ra)*t_val
        return face_number,r_int,rp[ray_ind]
#_______________________________________________________________________________
def compute_drag(az=30.,el=45.,Ny=20,Nz=20,yp=3.0,zp=3.0,rc=3.0) :
    rays = range((Ny+1)*(Nz+1))
    hits = []
    sources = []
    faces = []
    drag_mult = 1.0
    
    rp,rp_refl = ray_coords(az=az,el=el,yp=yp,zp=zp,Ny=Ny,Nz=Nz,rc=rc)
    ray_norm = rp_refl[0]-rp[0]
    ray_norm = ray_norm/norm(ray_norm)
    
    ray_tracer = Pool(cpu_count())
    results = ray_tracer.map(partial(compute_intersections,az=az,el=el,Ny=Ny,
                                     yp=yp,zp=zp,Nz=Nz,rc=rc),rays) 
    ray_tracer.close()
    
    for result in results :
        if result[0] != '' : 
            faces.append(result[0])
            hits.append(result[1])
            sources.append(result[2])
    hits = np.array(hits)
    sources = np.array(sources)
    force = drag_mult*ray_norm*hits.shape[0]
    diff_torques = np.cross(hits,drag_mult*ray_norm)
    torque = np.sum(diff_torques,axis=0)
    torque[abs(torque) < 1e-12] = 0.0
    return force,torque,hits,sources,faces
#_______________________________________________________________________________
def tabulate_drag(Ny,Nz,yp,zp,rc) :
    # Quantization of polar angles of flow
    Naz = 36
    Nel = 18
    drag_table = []
    print("Calculating drag force and torque for :")
    for az in np.arange(0,360+360/Naz,360/Naz) :
        temp = []
        for el in np.arange(-90,90+180/Nel,180/Nel) :
            stdout.write("\rAzimuth :"+str(az)+" deg, Elevation :"+str(el)+" deg")
            f,t,h,s,ff = compute_drag(az=az,el=el,Ny=Ny,Nz=Nz,yp=yp,zp=zp,rc=rc)
            temp.append((f,t))
        drag_table.append(temp)
    return drag_table
         


