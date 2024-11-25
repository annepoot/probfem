// This variable is a characteristic length and it controles the mesh size around a Point.
// It is possible to specify more than one variable for this purpose.
h = 0.5;

// These variables control the dimensions
L = 1.0;
H = 1.0;

// Points contains the x, y and z coordinate and the characteristic length of the Point.
Point(1) = {0,0,0,h};
Point(2) = {L,0,0,h};
Point(3) = {L,H,0,h};
Point(4) = {0,H,0,h};

// A Line is basically a connection between two Points. A good practice is to connect the
// Points in a counter-clockwise fashion.
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

// A Line Loop is the connection of Lines that defines an area. Again it is good practice
// to do this in a counter-clockwise fashion.
Line Loop(1) = {1,2,3,4};

// From the Line Loop it is now possible to create a surface, in this case a Plane Surface.
Plane Surface(1) = {1};

// From the Plane Surface a Physical Surface is generated, this makes is possible to only
// save elements which are defined on the area specified by the Line Loop.
Physical Surface(1) = {1};

Transfinite Curve {1, 2, 3, 4} = 3 Using Progression 1;
Transfinite Surface {1};
Recombine Surface {1};
