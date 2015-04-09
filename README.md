API description
===============

This library uses OpenGL to do spatial convolutions similarly to the Torch7 neural networks SpatialConvolution module.
The makefile also creates an executable for testing the underlying routines. It works under Linux and uses
GLX or EGL to initialize OpenGL.

It can use three different algorithms:

1) If the input batch is a multiple of 4, the input is put in a 3D
texture of size (`nInputPlanes, horz_tiles*width, vert_tiles*height`),
where `vert_tiles * horz_tiles * 4 = nBatchSize`. vert_tiles and
horz_tiles are automatically calculated. So the different images of the
input batch are tiled in the input texture making the texture bigger of
the size of the input image. Additionaly, images are also "tiled" in the
4 color planes of the texture. If n (from 0 to nBatchSize-1) is the
image index, they are put as indicated below:
for n%4 = 0, in the red color plane
for n%4 = 1, in the green color plane
for n%4 = 2, in the blu color plane
for n%4 = 3, in the alpha color plane
The filter is put in a uniform buffer of size `nInputPlanes*size*size`.
The output is put in a render buffer of size (`horz_tiles*width, 
vert_tiles*height`) and 4 GPU color planes. The render buffer will keep
the results of a complete calculation for one output plane.
The OpenGL drawing is started nOutputPlanes times, creating
nOutputPlanes different outputs. In each step, the filter input is
changed.

2) If the input batch is a multiple of 4 and the "use interpolator"
option is set, the input and the output are as in the algorithm above,
but filtering is performed in a different way. Since each texture lookup
in OpenGL can interpolate between four adjacent texels, this behaviour
can be exploited to do less explicit computations. The 4 texels
interpolator cannot be exploited completely, because this could only
be done in particular cases with filters that have special forms.
In the general case, only two adjacent texels can be fetched together
with the interpolator. Since the interpolator, in the 1D case, does
the operation `(1-a)*t1 + a*t2` (where 0<=a<=1 is the fetch point between
texels t1 and t2) and we have to do `t1*f1 + t2*f2` (f1 and f2 are our
filter coefficient), we can do the operation above with
`f*((1-a)*t1 + a*t2)` where `f=f1+f2 and a=f2/f`. This algorithm therefore
turns the filter coefficients in this second form and reduces the number
of operations in almost half. The uniform buffer used for the filters is
in this case long `3*nInputPlanes*floor((size*size+1)/2)*2`. There are
three coefficients for each point: the x coordinate of the texel, the
y coordinate of the texel and the filter coefficient. For example, in the
3x3 filter case, instead of having 5 filter coefficients, we reduce them
to 5, where coefficients 1 and 2 are reduced to a, 3 and 6 to b, 4 and 5
to c, 7 and 8 to d and 9 to e:

`1 2 3`

`4 5 6`

`7 8 9`

becomes

`a a b`

`c c b`

`d d e`            

Texel coordinates are not fixed anymore as in the previous algorithm, so
they are pre-calculated and then passed in the uniform buffer to the shader.

This is the fastest algorithm, but is a little bit less precise of the
other two. It's precision lowers as adjacent filter coefficients become
more and more close in absolute value, but opposite. If the are exactly
opposite, the algorithm fails, since the f above becomes 0 and therefore
a division by zero would occur.

3) If the input batch is not a multiple of 4, the input is put in a 3D
texture of size (`nInputPlanes/4, horz_tiles*width, vert_tiles*height`),
where `vert_tiles * horz_tiles = nBatchSize`. The different images of the
input batch are still tiled in the texture, but only there, they are not
"tiled" in the 4 available color planes. The input planes are divided
between the 4 available color planes and the depth of the texture. If
the input planes are less than 4, the other available color planes are
not used (wasting computation power).
The filter is put in another 3D texture of size (nInputPlanes/4,size,size)
and four GPU planes. The output is put in a render buffer of size
(`horz_tiles*width,vert_tiles*height`) and one GPU color plane. As before,
the render buffer will keep the results of a complete calculation for one
output plane and the OpenGL drawing is started nOutputPlanes times,
creating nOutputPlanes different outputs. In each step, the filter input
is changed. This algorithm is the slowest.

Both the input and output can be or byte (fixed Q2.6) or float.
Intermediate calculations are performed in the GPU with medium or high
precision.

## conv

Performs the convolution

Parameters:

- Input tensor (3D or 4D)
- Filter tensor (4D)
- Output tensor (3D or 4D)
- Bias tensor (optional, default all 0)
- Padding (optional, default 0)
- Step (optional, default 1)

Returns:

Nothing

## logging

Enables or disables logging (for debugging)

Parameters:

- 1 or 0

Returns:

Nothing
	
## clear

Flushes the shaders cache, clears the library status and closes OpenGL

Parameters:

Nothing

Returns:

Nothing
	
## precision

Sets the precision level

Parameters:

- Precision, which can be

- 0 byte precision (Q2.6) with internal calculations using medium float precision
- 1 byte precision (Q2.6) with internal calculations using high float precision
- 2 medium float precision
- 3 high float precision

Medium float precision normally means 16 bit floats, while high float precision normally
means 32 bit floats. This depends on the implementation in the GPU.

Returns:

Nothing
	
## useintp

Enables the use of the interpolator to speed up calculations

Parameters:

- 1 or 0

Returns:

Nothing
