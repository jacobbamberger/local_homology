import numpy as np
from scipy.spatial import distance_matrix
from gtda.homology import VietorisRipsPersistence

#this generates the data

#Q:should clouds overlap or not touch? 
#A:for nb neighbours it matters, for epsilon balls it does not matter too much 14/12JB 
#try some curvature?

def rotate_xy(theta, point_cloud):
    rotation=np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0,1]])
    return np.array([np.matmul(rotation, vec) for vec in point_cloud])

def rotate_yz(theta, point_cloud):
    rotation=np.array([[0, 0, 1], [0, np.cos(theta), np.sin(theta)], [0, -np.sin(theta), np.cos(theta)]])
    return np.array([np.matmul(rotation, vec) for vec in point_cloud])

def rotate_xz(theta, point_cloud):
    rotation=np.array([[0, 1, 0], [np.cos(theta), 0, np.sin(theta)], [-np.sin(theta),0, np.cos(theta)]])
    return np.array([np.matmul(rotation, vec) for vec in point_cloud])

def shift(vec, point_cloud):
    return np.array([np.add(vec, x) for x in point_cloud])

def generate_point_clouds(density=10):
	cube=np.array([[x, y, z] for x in np.arange(0,1, 1/density) for y in np.arange(0,1, 1/density) for z in np.arange(0,1, 1/density)])
	x_zero_plane=np.array([[0, y, z] for y in np.arange(0,np.sqrt(1.9), 1/density) for z in np.arange(0,1, 1/density)])
	x_neg_plane=np.array([[-1/2, y, z] for y in np.arange(0,np.sqrt(1.9), 1/density) for z in np.arange(0,1, 1/density)])
	very_high_x_quarter_plane=np.array([[1/4, y, z] for y in np.arange(0,1, 1/density) for z in np.arange(1.5, 2.5, 1/density)])

	top_plane_xhalf=np.array([[1/2, y, z] for y in np.arange(0,1, 1/density) for z in np.arange(1,2, 1/density)])
	top_plane_slanted=np.array([np.add(vec, [0,0,1]) for vec in rotate_xy(-np.pi/4,rotate_xz(np.pi/4,x_zero_plane))])
	parallel_plane=np.array([[x, y, 1.9] for x in np.arange(0,1, 1/density) for y in np.arange(0,1, 1/density)])

	corner_plane=shift([1/2*np.sqrt(2), -1/2*np.sqrt(2), -1/2], top_plane_slanted)
	corner_line=np.array([[0.9+t, t, 0.9+t] for t in np.arange(0, 1, 1/density)]) #try +1? or go to density +1

	side_perp_line=np.array([[x, 1/2, 1/2] for x in np.arange(-1,0, 1/density)])
	slanted_side_line=np.array([[x, 1/2-3/4*x, 1/2] for x in np.arange(-1,0, 1/density)])

	mega=list(cube)+list(top_plane_xhalf)+list(top_plane_slanted)+list(corner_plane)+list(corner_line)+list(side_perp_line)+list(slanted_side_line)+list(parallel_plane)+list(x_neg_plane)+list(very_high_x_quarter_plane)


    
	line_3d=[[x, 0, 0] for x in np.arange(-1,1, 1/density)]
	scnd_line=[[0, y, 0] for y in np.arange(-1,1, 1/density)]
	third_line=[[x, 1, 0] for x in np.arange(-1,1, 1/density)]
	lines=line_3d+scnd_line+third_line #THIS
	plane_line=np.array(list(x_neg_plane)+ list(side_perp_line)+list(slanted_side_line)) #THIS
	planes=np.array(list(x_zero_plane)+list(very_high_x_quarter_plane)+list(top_plane_xhalf)+list(top_plane_slanted)+list(parallel_plane)+list(corner_plane))
	simple_planes=[]
	for x in lines:
		x=np.array(x)
		simple_planes+=[x]
		simple_planes+=[x+np.array([0, 0, i]) for i in np.arange(0.1, 1, 0.1)]
	simple_planes=np.array(simple_planes)
    
	line=np.array([[x, 0, 0] for x in np.arange(0, 1, 0.1)])
	plane=np.array([[x, y, 0] for x in np.arange(0, 1, 0.1) for y in np.arange(0, 1, 0.1)])
	cube=np.array([[x, y, z] for x in np.arange(0, 1, 0.1) for y in np.arange(0, 1, 0.1) for z in np.arange(0, 1, 0.1)])
    
    
#just the surface...
	density = 1/10
	cloud=[]
	for x in np.arange(-1, 1, density):
		up = np.sqrt(2**2 - (x-1)**2)
		bot = np.sqrt(1-x**2)
		cy = (bot+up)/2

		for y in np.arange(bot+0.001, up, density):
			#for z in np.arange(density, np.sqrt((up-cy)**2-(y-cy)**2), density):
			z = np.sqrt(((up-bot)/2)**2-(y-cy)**2)
			cloud+=[ [x, y, z], [x, -y, z], [x, y, -z], [x, -y, -z]]

	for x in np.arange(1, 3, density):
		up=np.sqrt(2**2 - (x-1)**2)
		bot=-np.sqrt(2**2 - (x-1)**2)
		for y in np.arange(0, up, density):
			#for z in np.arange(density, np.sqrt(up**2 - y**2)/2, density):
			z = np.sqrt(up**2 - y**2)/2
			cloud+=[ [x, y, z], [x, -y, z], [x, y, -z], [x, -y, -z]]
        

	cloud+=[[-1, 0, 0]]

	x=-1+density/5
	y=(np.sqrt(2**2 - (x-1)**2)+np.sqrt(1-x**2))/2
	cloud+=[[x, y, 0], [x, -y, 0]]

	x=-1+density/1.4
	y=(np.sqrt(2**2 - (x-1)**2)+np.sqrt(1-x**2))/2
	cloud+=[[x, y, 0], [x, -y, 0]]

	x=-1+density/0.5
	y=(np.sqrt(2**2 - (x-1)**2)+np.sqrt(1-x**2))/2
	cloud+=[[x, y, 0], [x, -y, 0]]

	cloud = np.array(cloud)
    
	line = np.array([[0, 0, z] for z in np.arange(0, 1, 1/10)])
	plane = np.array([[0, y, z] for z in np.arange(0, 1, 1/10) for y in np.arange(0, 1, 1/10)])
	cube = np.array([[x, y, z] for x in np.arange(0, 1, 1/10) for y in np.arange(0, 1, 1/10) for z in np.arange(0, 1, 1/10)])

	line_plane_cube = np.append(np.append(line, [p+[0.5, -0.5, 0] for p in plane], axis=0), [p+[1, -0.5 ,0] for p in cube], axis=0)



	return line, plane, cube, lines, line_plane_cube, cloud #simple_planes, planes, plane_line, mega, 


###