#include <luaT.h>
#include <TH/TH.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <locale.h>
#include <sys/time.h>
#include <pthread.h>
#ifdef USEOMP
#	include <omp.h>
#endif
#ifdef USEGLX
#	define GL_GLEXT_PROTOTYPES
#	include <X11/Xlib.h>
#	include <GL/glx.h>
#	include <GL/gl.h>
#	include <GL/glext.h>
static GLXContext ctx;
static Window win;
static Display *dpy;
#else
#	include <EGL/egl.h>
#	ifdef USEGLES2
#		include <GLES2/gl2.h>
#	else
#		include <GLES3/gl3.h>
#	endif
static EGLDisplay display;
static EGLContext context;
static EGLSurface surface;
#endif

// Zero is different on different architectures
// It's 127 for NVidia PC and 128 for ARM Mali
#ifdef USEGLX
#define UZERO 127
#else
#define UZERO 128
#endif

enum {OUT_DEFAULT, OUT_BYTE, OUT_INTEGER, OUT_FLOAT};

static int loops = 1, simplefilter, simpleinput, nocheck, printoutput;
#ifndef NOLIB
static int useinterpolator, precision;
#endif
static double times[3];
static double Gops;
static FILE *fpbatch;
static char sresult[300];
static char logging;
static GLuint programs_cache[20][2];

double seconds()
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}

void lprintf(const char *fmt, ...)
{
	if(!logging)
		return;
	char s[300];
	static double start;
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(s, sizeof(s), fmt, ap);
	if(!start)
		start = seconds();
	printf("%f: %s", seconds() - start, s);
	va_end(ap);
}

void eprintf(const char *fmt, ...)
{
	char s[300];
	static double start;
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(s, sizeof(s), fmt, ap);
	if(!start)
		start = seconds();
	printf("%f: %s", seconds() - start, s);
	va_end(ap);
}

typedef struct
{
	int nimages;
	int nplanes;
	int height;
	int width;
	int padding;
	float *data;
} INPUTS;

typedef struct
{
	int nfilters;
	int nplanes;
	int size;
	int step;
	float *data;
	float *bias;
} FILTERS;

typedef struct
{
	int nimages;
	int nfilters;
	int height;
	int width;
	float *data;
} OUTPUTS;

typedef struct
{
	GLuint renderbuffer, framebuffer;
	GLuint programObject;
	GLint positionLoc;
	GLint texCoordLoc;
	GLint sInput;	// Input texture sampler location
	GLint sFilter;	// Filter texture sampler location
	GLuint tId;	// Input texture ID
	GLuint ftId;	// Filter texture ID
	int fsize;		// Filter size
	int mh, mw;		// Multiplier for each dimension: how many inputs are tiled in each direction
	int h, w;		// Tile size
	int th, tw;		// Texture size
	int oth, otw;	// Output render buffer size
	int oh, ow;		// Output tile size
	int nplanes;	// Number of input planes
	int pstride;	// Input plane stride
	int stride;		// Input line stride
	int ostride;	// Output line stride
	int outtype;	// Render buffer type (OUT_... enums)
	char floattexture;	// Input texture type (1 if float, 0 if byte)
	int quadinputs;	// If input images are multiple of 4, use another, faster tiling 
	int highp;		// Use highp instead of medium precision
	int usebuffer;	// Use buffer instead of texture for the filter
	int useinterpolator;	// Process two points at the same time exploiting the interpolator
} GLSTATUS;

INPUTS inputs;
FILTERS filters;
OUTPUTS outputs, outputs_cpu;
GLSTATUS gls;

void *cmalloc(size_t nbytes)
{
	void *buf = malloc(nbytes);
	if(buf)
		return buf;
	eprintf("malloc of %u bytes failed\n", nbytes);
	return 0;
}

void RandomInputs(INPUTS *in)
{
	int i, n = in->nimages * in->nplanes * in->width * in->height;
	in->data = (float *)cmalloc(n * sizeof(float));
	for(i = 0; i < n; i++)
		in->data[i] = (float)rand() / RAND_MAX;
}

void RandomFilters(FILTERS *f)
{
	int i, n = f->nfilters * f->nplanes * f->size * f->size;
	f->data = (float *)cmalloc(n * sizeof(float));
	for(i = 0; i < n; i++)
		f->data[i] = (float)rand() / (RAND_MAX * (double)n);
}

void SimpleInputs(INPUTS *in)
{
	int n = in->nimages * in->nplanes * in->width * in->height;
	in->data = (float *)calloc(n, sizeof(float));
	in->data[2*in->width + 2] = -0.5f;
	//in->data[0*in->width + 9] = 0.5f;
}

void SimpleFilters(FILTERS *f)
{
	int n = f->nfilters * f->nplanes * f->size * f->size;
	f->data = (float *)calloc(n, sizeof(float));
	f->data[0] = 0.1f;
	f->data[1] = 0.01f;
	int i;
	for(i = 0; i < n; i++)
		f->data[i] = 0.1f;
}

void AllocOutput(INPUTS *in, FILTERS *f, OUTPUTS *out)
{
	out->data = (float *)cmalloc(out->nimages * out->nfilters * out->height * out->width * sizeof(float));
}

void Convolve(INPUTS *in, FILTERS *f, OUTPUTS *out)
{
	int i, j, k, h, plane, image, filter;
	float tmp;

	AllocOutput(in, f, out);
#ifdef USEOMP
#pragma omp parallel for private (filter, image, i, j, tmp, plane, k, h)
#endif
	for(image = 0; image < in->nimages; image++)
		for(filter = 0; filter < f->nfilters; filter++)
			for(i = 0; i < out->height; i++)
				for(j = 0; j < out->width; j++)
				{
					tmp = 0;
					for(plane = 0; plane < in->nplanes; plane++)
					{
						for(k = 0; k < f->size; k++)
							for(h = 0; h < f->size; h++)
								tmp += in->data[((image * in->nplanes + plane) * in->height + (i*f->step + k)) * in->width + j*f->step + h] *
									f->data[((filter * f->nplanes + plane) * f->size + k) * f->size + h];
					}
					out->data[((image * out->nfilters + filter) * out->height + i) * out->width + j] = tmp;
				}
}

void MeasureError(OUTPUTS *out1, OUTPUTS *out2, int filtersize)
{
	int i, j, image, filter;
	float tmp, error = 0;

	for(image = 0; image < out1->nimages; image++)
		for(filter = 0; filter < out1->nfilters; filter++)
			for(i = 0; i < out1->height; i++)
				for(j = 0; j < out1->width; j++)
				{
					if(printoutput == 2)
						lprintf("(%d,%d,%d,%d) %f %f\n", image, filter, i, j,
							out1->data[((image * out1->nfilters + filter) * out1->height + i) * out1->width + j],
							out2->data[((image * out1->nfilters + filter) * out1->height + i) * out1->width + j]);
					tmp = out1->data[((image * out1->nfilters + filter) * out1->height + i) * out1->width + j] -
						  out2->data[((image * out1->nfilters + filter) * out1->height + i) * out1->width + j];
					tmp = fabs(tmp);
					if(tmp > error)
						error = tmp;
				}
	sprintf(sresult + strlen(sresult), "%f\n", error);
}

void PrintNonZero(OUTPUTS *out, const char *desc)
{
	int i, j, filter, image;
	
	for(image = 0; image < out->nimages; image++)
		for(filter = 0; filter < out->nfilters; filter++)
			for(i = 0; i < out->height; i++)
				for(j = 0; j < out->width; j++)
					if(fabs(out->data[((image * out->nfilters + filter) * out->height + i) * out->width + j]) > 0.000001)
						lprintf("%s(%d,%d,%d,%d) = %f\n", desc, image, filter, i, j,
							out->data[((image * out->nfilters + filter) * out->height + i) * out->width + j]);
}

int checkGlError(const char *op)
{
	int rc = 0, error;
	while ((error = glGetError()) != GL_NO_ERROR) {
		eprintf("%s: Error 0x%x\n", op, error);
		rc = error;
	}
	return rc;
}

void SetByteInput(GLSTATUS *gls, INPUTS *in, GLbyte *texture)
{
	int plane, mh, mw, y, x, c, colors;
		
	// Tile the batch inside the texture

	colors = in->nplanes < 4 ? in->nplanes : 4;
#ifdef USEOMP
#pragma omp parallel for private (plane, mh, y, mw, x, c)
#endif
	for(plane = 0; plane < gls->nplanes; plane++)
		for(mh = 0; mh < gls->mh; mh++)
			for(y = 0; y < in->height; y++)
				for(mw = 0; mw < gls->mw; mw++)
					for(x = 0; x < in->width; x++)
						for(c = 0; c < colors; c++)
							texture[plane * gls->pstride + (mh * gls->h + in->padding + y) * gls->stride + (mw * gls->w + in->padding + x) * 4 + c] =
								(GLbyte)(in->data[((((mh * gls->mw + mw) * gls->nplanes + plane) * colors + c) * in->height + y) * in->width + x] * 64.0f);
}

void SetByteInputQuadInputs(GLSTATUS *gls, INPUTS *in, GLbyte *texture)
{
	int plane, mh, mw, y, x, c;
		
	// Tile the batch inside the texture
#ifdef USEOMP
#pragma omp parallel for private (plane, mh, y, mw, x, c)
#endif
	for(plane = 0; plane < in->nplanes; plane++)
		for(mh = 0; mh < gls->mh; mh++)
			for(y = 0; y < in->height; y++)
				for(mw = 0; mw < gls->mw; mw++)
					for(x = 0; x < in->width; x++)
						for(c = 0; c < 4; c++)
							texture[plane * gls->pstride +
								(mh * gls->h + in->padding + y) * gls->stride +
								(mw * gls->w + in->padding + x) * 4 + c] =
								(GLbyte)(in->data[((((mh * gls->mw + mw) * 4 + c) *
								in->nplanes + plane) * in->height + y) * in->width + x] * 64.0f);
}

int CreateTextureByteInput(GLSTATUS *gls, INPUTS *in)
{
	GLbyte *texture;
	
	// Input will be in texture 0
	glActiveTexture(GL_TEXTURE0);
	if(checkGlError("glActiveTexture"))
		return -1;

	// Set the input sampler to texture 0
	glUniform1i(gls->sInput, 0);
	if(checkGlError("glUniformInt"))
		return -1;

	lprintf("Creating input texture\n");
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	glGenTextures(1, &gls->tId);
	glBindTexture(GL_TEXTURE_3D, gls->tId);
	if(checkGlError("glBindTexture"))
		return -1;

	texture = (GLbyte *)cmalloc(gls->nplanes * gls->tw * gls->th * 4);
	if((gls->nplanes < 4 && !gls->quadinputs) || in->padding)
		memset(texture, 0, gls->nplanes * gls->tw * gls->th * 4);
	if(gls->quadinputs)
		SetByteInputQuadInputs(gls, in, texture);
	else SetByteInput(gls, in, texture);
	if(printoutput == 3)
	{
		int i;
		
		for(i = 0; i < gls->th * gls->tw * gls->nplanes * 4; i++)
			if(texture[i])
				lprintf("in[%x]=%x\n", i, texture[i]);
	}
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8_SNORM, gls->tw, gls->th, gls->nplanes,
		0, GL_RGBA, GL_BYTE, texture);
	if(checkGlError("glTexImage3D"))
		return -1;
	free(texture);

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, gls->useinterpolator ? GL_LINEAR : GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, gls->useinterpolator ? GL_LINEAR : GL_NEAREST);
	return 0;
}

void SetFloatInput(GLSTATUS *gls, INPUTS *in, GLfloat *texture)
{
	int plane, mh, mw, y, x, c, colors;
		
	colors = in->nplanes < 4 ? in->nplanes : 4;
	// Tile the batch inside the texture
#ifdef USEOMP
#pragma omp parallel for private (plane, mh, y, mw, x, c)
#endif
	for(plane = 0; plane < gls->nplanes; plane++)
		for(mh = 0; mh < gls->mh; mh++)
			for(y = 0; y < in->height; y++)
				for(mw = 0; mw < gls->mw; mw++)
					for(x = 0; x < in->width; x++)
						for(c = 0; c < colors; c++)
							texture[plane * gls->pstride + (mh * gls->h + in->padding + y) * gls->stride + (mw * gls->w + in->padding + x) * 4 + c] =
								in->data[((((mh * gls->mw + mw) * gls->nplanes + plane) * colors + c) * in->height + y) * in->width + x];
}

void SetFloatInputQuadInputs(GLSTATUS *gls, INPUTS *in, GLfloat *texture)
{
	int plane, mh, mw, y, x, c;
		
	// Tile the batch inside the texture
#ifdef USEOMP
#pragma omp parallel for private (plane, mh, y, mw, x, c)
#endif
	for(plane = 0; plane < in->nplanes; plane++)
		for(mh = 0; mh < gls->mh; mh++)
			for(y = 0; y < in->height; y++)
				for(mw = 0; mw < gls->mw; mw++)
					for(x = 0; x < in->width; x++)
						for(c = 0; c < 4; c++)
							texture[plane * gls->pstride +
								(mh * gls->h + in->padding + y) * gls->stride +
								(mw * gls->w + in->padding + x) * 4 + c] =
								in->data[((((mh * gls->mw + mw) * 4 + c) *
								in->nplanes + plane) * in->height + y) * in->width + x];
}

int CreateTextureFloatInput(GLSTATUS *gls, INPUTS *in)
{
	GLfloat *texture;
	
	// Input will be in texture 0
	glActiveTexture(GL_TEXTURE0);
	if(checkGlError("glActiveTexture"))
		return -1;

	// Set the input sampler to texture 0
	glUniform1i(gls->sInput, 0);
	if(checkGlError("glUniformInt"))
		return -1;

	lprintf("Creating input texture\n");
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	glGenTextures(1, &gls->tId);
	glBindTexture(GL_TEXTURE_3D, gls->tId);
	if(checkGlError("glBindTexture"))
		return -1;

	texture = (GLfloat *)cmalloc(gls->nplanes * gls->tw * gls->th * 4 * sizeof(float));
	if((gls->nplanes < 4 && !gls->quadinputs) || in->padding)
		memset(texture, 0, gls->nplanes * gls->tw * gls->th * 4 * sizeof(float));
	if(gls->quadinputs)
		SetFloatInputQuadInputs(gls, in, texture);
	else SetFloatInput(gls, in, texture);
	if(printoutput == 3)
	{
		int i;
		
		for(i = 0; i < gls->th * gls->tw * gls->nplanes * 4; i++)
			if(texture[i])
				lprintf("in[%x]=%f\n", i, texture[i]);
	}
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, gls->tw, gls->th, gls->nplanes,
		0, GL_RGBA, GL_FLOAT, texture);
	if(checkGlError("glTexImage3D"))
		return -1;
	free(texture);

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, gls->useinterpolator ? GL_LINEAR : GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, gls->useinterpolator ? GL_LINEAR : GL_NEAREST);
	return 0;
}

void SetFloatFilters(FILTERS *f, float *texture, int filter)
{
	int i, j, plane, c, nplanes = (f->nplanes+3) / 4, colors;
	
	colors = f->nplanes < 4 ? f->nplanes : 4;
	if(colors < 4)
		memset(texture, 0, 4 * f->size * f->size * sizeof(float));
	for(plane = 0; plane < nplanes; plane++)
		for(i = 0; i < f->size; i++)
			for(j = 0; j < f->size; j++)
				for(c = 0; c < colors; c++)
					texture[(((plane * f->size) + i) * f->size + j) * 4 + c] =
						f->data[(((filter * f->nplanes + colors * plane + c) * f->size) + i) * f->size + j];
}

void SetFloatFiltersQuadInputs(FILTERS *f, float *texture, int filter)
{
	int i, j, plane;
	
	for(plane = 0; plane < f->nplanes; plane++)
		for(i = 0; i < f->size; i++)
			for(j = 0; j < f->size; j++)
				texture[((plane * f->size) + i) * f->size + j] =
					f->data[(((filter * f->nplanes + plane) * f->size) + i) * f->size + j];
}

void SetFloatFiltersQuadInputsInterpolator(GLSTATUS *gls, FILTERS *f, float *texture, int filter)
{
	int i, j, plane, n = 0;
	float xstep = 1.0 / gls->tw;
	float ystep = 1.0 / gls->th;
	float fsum, a, f1, f2;
	
	for(plane = 0; plane < f->nplanes; plane++)
		for(i = 0; i < f->size; i++)
			for(j = 0; j < f->size; j += 2)
			{
				if(j + 1 == f->size)
				{
					// Last column, interpolate vertically
					if(i % 2 == 0)
					{
						if(i == f->size - 1 && j == f->size - 1)
						{
							// Last, no interpolation
							texture[n++] = j * xstep;
							texture[n++] = j * ystep;
							texture[n++] = f1 = f->data[(((filter * f->nplanes + plane) * f->size) + i) * f->size + j];
						} else {
							f1 = f->data[(((filter * f->nplanes + plane) * f->size) + i) * f->size + j];
							f2 = f->data[(((filter * f->nplanes + plane) * f->size) + i + 1) * f->size + j];
							fsum = f1+f2;
							if(!fsum)
								a = 0;
							else a = f2/fsum;
							texture[n++] = j * xstep;
							texture[n++] = (i + a) * ystep;
							texture[n++] = fsum;
						}
					} // else skip, we already took this element
				} else {
					f1 = f->data[(((filter * f->nplanes + plane) * f->size) + i) * f->size + j];
					f2 = f->data[(((filter * f->nplanes + plane) * f->size) + i) * f->size + j + 1];
					fsum = f1+f2;
					if(!fsum)
						a = 0;
					else a = f2/fsum;
					texture[n++] = (j + a) * xstep;
					texture[n++] = i * ystep;
					texture[n++] = fsum;
				}
			}
}

int CreateFilterFloatInputBuffer(GLSTATUS *gls, FILTERS *f, int filter)
{
	size_t alloc = gls->useinterpolator ?
		(f->size * f->size + 1) / 2 * 3 * f->nplanes * sizeof(float) :
		f->size * f->size * f->nplanes * sizeof(float);
	glGenBuffers(1, &gls->ftId);
	if(checkGlError("glGenBuffers"))
		return -1;
	glBindBuffer( GL_UNIFORM_BUFFER, gls->ftId);
	if(checkGlError("glBindBuffer"))
		return -1;
	if(gls->quadinputs)
	{
		float *texture = (float *)cmalloc(alloc);
		if(gls->useinterpolator)
			SetFloatFiltersQuadInputsInterpolator(gls, f, texture, filter);
		else SetFloatFiltersQuadInputs(f, texture, filter);
		GLuint blockId = glGetUniformBlockIndex(gls->programObject, "filterblock");
		if(blockId == -1)
		{
			eprintf("blockID is 0\n");
			return -1;
		}
		glUniformBlockBinding (gls->programObject, blockId, 1);
		if(checkGlError("glUnifromBlockBinding"))
			return -1;
		if(gls->useinterpolator)
			glBufferData (GL_UNIFORM_BUFFER, alloc, texture, GL_DYNAMIC_DRAW);
		else glBufferData (GL_UNIFORM_BUFFER, alloc, texture, GL_DYNAMIC_DRAW);
		if(checkGlError("glBufferData"))
			return -1;
		glBindBufferBase(GL_UNIFORM_BUFFER, 1, gls->ftId);
		if(checkGlError("glBindBufferBase"))
			return -1;
		free(texture);
		return 0;
	} else {
		eprintf("Mode of operation not supported\n");
		return -1;
	}
}

int CreateFilterFloatInput(GLSTATUS *gls, FILTERS *f, int filter)
{
	float *texture;
	
	//lprintf("Creating filter texture\n");

	// Filter will be in texture1
	glActiveTexture(GL_TEXTURE1);
	if(checkGlError("glActiveTexture"))
		return -1;
	// Set the filter sampler to texture 1
	glUniform1i(gls->sFilter, 1);
	if(checkGlError("glUniformInt"))
		return -1;

	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	glGenTextures(1, &gls->ftId);

	glBindTexture(GL_TEXTURE_3D, gls->ftId);
	if(checkGlError("glBindTexture"))
		return -1;

	if(gls->quadinputs)
	{
		texture = (float *)cmalloc(f->size * f->size * f->nplanes * sizeof(float));
		SetFloatFiltersQuadInputs(f, texture, filter);
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, f->size, f->size, f->nplanes,
			0, GL_RED, GL_FLOAT, texture);
	} else {
		texture = (float *)cmalloc(f->size * f->size * gls->nplanes * 4 * sizeof(float));
		SetFloatFilters(f, texture, filter);
		glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, f->size, f->size, gls->nplanes,
			0, GL_RGBA, GL_FLOAT, texture);
	}
	free(texture);
	if(checkGlError("glTexImage3D"))
		return -1;

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	return 0;
}

int ReadFrameByte(GLSTATUS *gls, OUTPUTS *out, int filter)
{
	GLubyte *frame;
	int mh, mw, y, x, stride;
	
	glBindFramebuffer(GL_FRAMEBUFFER, gls->framebuffer);
	if(checkGlError("glBindFramebuffer"))
		return -1;

	//glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &n);
	//lprintf("GL_IMPLEMENTATION_COLOR_READ_FORMAT = %x\n", n);
	stride = (gls->otw + 3) / 4 * 4;
	frame = (GLubyte *)cmalloc(gls->oth * stride);
	glReadPixels(0, 0, gls->otw, gls->oth, GL_RED, GL_UNSIGNED_BYTE, frame);
	if(checkGlError("glReadPixels"))
		return -1;
	if(printoutput == 3)
	{
		int i;
		
		for(i = 0; i < gls->oth * stride; i++)
			if(frame[i])
				lprintf("out[%x]=%x\n", i, frame[i]);
	}
	// Untile the batch inside the texture
#ifdef USEOMP
#pragma omp parallel for private (y, mh, mw, x)
#endif
	for(y = 0; y < out->height; y++)
		for(mh = 0; mh < gls->mh; mh++)
			for(mw = 0; mw < gls->mw; mw++)
				for(x = 0; x < out->width; x++)
					out->data[(((mh * gls->mw + mw) * out->nfilters + filter) * out->height + y) * out->width + x] =
						(frame[(mh * gls->oh + y) * stride + mw * gls->ow + x]-UZERO) * (1.0f/64.0f);
	free(frame);
	return 0;
}

int ReadFrameFloat(GLSTATUS *gls, OUTPUTS *out, int filter)
{
	GLfloat *frame;
	int mh, mw, y, x;
	
	glBindFramebuffer(GL_FRAMEBUFFER, gls->framebuffer);
	if(checkGlError("glBindFramebuffer"))
		return -1;

//	glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &n);
//	lprintf("GL_IMPLEMENTATION_COLOR_READ_FORMAT = %x\n", n);
	frame = (GLfloat *)cmalloc(gls->oth * gls->otw * sizeof(float));
	if(gls->outtype == OUT_INTEGER)
		glReadPixels(0, 0, gls->otw, gls->oth, GL_RED_INTEGER, GL_INT, frame);
	else glReadPixels(0, 0, gls->otw, gls->oth, GL_RED, GL_FLOAT, frame);
	if(checkGlError("glReadPixels"))
		return -1;
	if(printoutput == 3)
	{
		int i;
		
		for(i = 0; i < gls->oth * gls->otw; i++)
			if(frame[i])
				lprintf("out[%x]=%f\n", i, frame[i]);
	}
	// Untile the batch inside the texture
#ifdef USEOMP
#pragma omp parallel for private (y, mh, mw, x)
#endif
	for(y = 0; y < out->height; y++)
		for(mh = 0; mh < gls->mh; mh++)
			for(mw = 0; mw < gls->mw; mw++)
				for(x = 0; x < out->width; x++)
					out->data[(((mh * gls->mw + mw) * out->nfilters + filter) * out->height + y) * out->width + x] =
						frame[((mh * gls->oh + y) * gls->mw + mw) * gls->ow + x];
	free(frame);
	return 0;
}

int ReadFrameByteQuadInputs(GLSTATUS *gls, OUTPUTS *out, int filter)
{
	GLubyte *frame;
	int mh, mw, y, x, c;
	
	glBindFramebuffer(GL_FRAMEBUFFER, gls->framebuffer);
	if(checkGlError("glBindFramebuffer"))
		return -1;

	frame = (GLubyte *)cmalloc(gls->oth * gls->otw * 4);
	glReadPixels(0, 0, gls->otw, gls->oth, GL_RGBA, GL_UNSIGNED_BYTE, frame);
	if(checkGlError("glReadPixels"))
		return -1;
	if(printoutput == 3)
	{
		int i;
		
		for(i = 0; i < gls->oth * gls->otw * 4; i++)
			if(frame[i])
				lprintf("out[%x]=%x\n", i, frame[i]);
	}
	// Untile the batch inside the texture
#ifdef USEOMP
#pragma omp parallel for private (mh, y, mw, x, c)
#endif
	for(mh = 0; mh < gls->mh; mh++)
		for(y = 0; y < out->height; y++)
			for(mw = 0; mw < gls->mw; mw++)
				for(x = 0; x < out->width; x++)
					for(c = 0; c < 4; c++)
						out->data[((((mh * gls->mw + mw) * 4 + c) * out->nfilters + filter) * out->height + y) * out->width + x] =
							(frame[(mh * gls->oh + y) * gls->ostride + (mw * gls->ow + x) * 4 + c]-UZERO) * (1.0f/64.0f);
	free(frame);
	return 0;
}

int ReadFrameFloatQuadInputs(GLSTATUS *gls, OUTPUTS *out, int filter)
{
	GLfloat *frame;
	int mh, mw, y, x, c;
	
	glBindFramebuffer(GL_FRAMEBUFFER, gls->framebuffer);
	if(checkGlError("glBindFramebuffer"))
		return -1;

	frame = (GLfloat *)cmalloc(gls->oth * gls->otw * 4 * sizeof(float));
	if(gls->outtype == OUT_INTEGER)
		glReadPixels(0, 0, gls->otw, gls->oth, GL_RGBA_INTEGER, GL_INT, frame);
	else glReadPixels(0, 0, gls->otw, gls->oth, GL_RGBA, GL_FLOAT, frame);
	if(checkGlError("glReadPixels"))
		return -1;
	if(printoutput == 3)
	{
		int i;
		
		for(i = 0; i < gls->oth * gls->otw * 4; i++)
			if(frame[i])
				lprintf("out[%x]=%f\n", i, frame[i]);
	}
	// Untile the batch inside the texture
#ifdef USEOMP
#pragma omp parallel for private (mh, y, mw, x, c)
#endif
	for(mh = 0; mh < gls->mh; mh++)
		for(y = 0; y < out->height; y++)
			for(mw = 0; mw < gls->mw; mw++)
				for(x = 0; x < out->width; x++)
					for(c = 0; c < 4; c++)
						out->data[((((mh * gls->mw + mw) * 4 + c) * out->nfilters + filter) * out->height + y) * out->width + x] =
							frame[(mh * gls->oh + y) * gls->ostride + (mw * gls->ow + x) * 4 + c];
	free(frame);
	return 0;
}

const char *mVertexShader =
	"#version 300 es\n"
	"in vec4 aPosition;\n"
	"in vec2 aTextureCoord;\n"
	"out vec3 vTextureCoord;\n"
	"void main() {\n"
	"gl_Position = aPosition;\n"
	"vTextureCoord = vec3(aTextureCoord, 0.0);\n"
	"}\n";

char *CreateShaderSource(GLSTATUS *gls)
{
	const char *begin =
	"#version 300 es\n"
	"precision %s float;\n"
	"in vec3 vTextureCoord;\n"
	"out %s FragColor;\n"
	"float z;\n"
	"uniform float bias;\n"
	"uniform float xstep;\n"
	"uniform float ystep;\n"
	"uniform float pstep;\n"
	"uniform %s sampler3D sInput;\n";
	const char *mainbegin = "void main() {\n"
		"vec4 sum = vec4(%s);\n"
		"for(z = pstep*0.5; z < 1.0; z += pstep) {\n"
		"sum += ";
	const char *mainbegin2 = "void main() {\n"
		"vec4 sum = vec4(bias);\n"
		"int idx = 0;\n"
		"for(z = pstep*0.5; z < 1.0; z += pstep) {\n"
		"sum += ";
	const char *end =
		";\n"
		"}\n"
		"FragColor = (bias + sum.r + sum.g + sum.b + sum.a)%s;\n"
	"}\n";
	const char *endqi =
		";\n"
		"}\n"
		"FragColor = sum%s;\n"
	"}\n";
		
	const char *addline = "texture(sInput,vTextureCoord+vec3(%d.0*xstep, %d.0*%s, z)) * texture(sFilter, vec3(%1.3f,%1.3f, z))";
	const char *addline2 = "texture(sInput,vTextureCoord+vec3(%d.0*xstep, %d.0*%s, z)) * filterdata[idx + %d]\n";
	const char *addline3 = "texture(sInput,vTextureCoord+vec3(filterdata[idx+%d], filterdata[idx+%d], z)) * filterdata[idx + %d]\n";
	char *shader, *ps;
	int i, j, size = gls->fsize;
	shader = (char *)malloc(strlen(begin) + size * size * (strlen(addline)+8) + strlen(end)+500);
	sprintf(shader, begin, gls->highp ? "highp" : "mediump",
		gls->quadinputs ? "vec4" : "float",
		gls->highp ? "highp" : "mediump");
	if(gls->usebuffer)
	{
		sprintf(shader + strlen(shader), "uniform filterblock { float filterdata[%d]; };\n", size * size * gls->nplanes);
		strcat(shader, mainbegin2);
	} else {
		sprintf(shader + strlen(shader), "uniform %s sampler3D sFilter;\n", gls->highp ? "highp" : "mediump");
		sprintf(shader + strlen(shader), mainbegin, gls->quadinputs ? "bias" : "0.0");
	}
	ps = shader + strlen(shader);
	if(gls->useinterpolator)
	{
		j = 0;
		for(i = 0; i < size*size; i += 2)
		{
			if(i)
			{
				strcpy(ps, " + ");
				ps += 3;
			}
			sprintf(ps, addline3, j, j+1, j+2);
			j += 3;
			ps += strlen(ps);
		}
		sprintf(ps, ";\nidx += %d", j);
		ps += strlen(ps);	
	} else {
		for(i = 0; i < size; i++)
			for(j = 0; j < size; j++)
			{
				if(i || j)
				{
					strcpy(ps, " + ");
					ps += 3;
				}
				if(gls->usebuffer)
				{
					sprintf(ps, addline2, j, i, gls->tw == gls->th ? "xstep" : "ystep", j + i *size);
				} else {
					sprintf(ps, addline, j, i, gls->tw == gls->th ? "xstep" : "ystep", (j+0.5)/size, (i+0.5)/size);
					strcat(ps, gls->quadinputs ? ".r\n" : "\n");
				}
				ps += strlen(ps);
			}
		if(gls->usebuffer)
		{
			sprintf(ps, ";\nidx += %d", size * size);
			ps += strlen(ps);
		}
	}
	sprintf(ps, gls->quadinputs ? endqi : end,
		gls->outtype == OUT_BYTE && !gls->floattexture ? "*0.5 + 0.5" :
		gls->outtype == OUT_BYTE && gls->floattexture ? "*0.25 + 0.5" :
		gls->outtype != OUT_BYTE && gls->floattexture ? "" :
		"*2.0");
	return shader;	
}

int InitGL(int w, int h)
{
#ifdef USEGLX
	static int attributeList[] = { GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None };
	dpy = XOpenDisplay(0);

	if(!dpy)
	{
		lprintf("Cannot open display, trying :0\n");
		dpy = XOpenDisplay(":0");
		if(!dpy)
		{
			eprintf("Cannot open display\n");
			return -1;
		}
	}
	XVisualInfo *vi = glXChooseVisual(dpy, DefaultScreen(dpy), attributeList);
	XSetWindowAttributes swa;
	swa.colormap = XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, AllocNone);
	swa.border_pixel = 0;
	swa.event_mask = StructureNotifyMask;
	win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 0, 0, w, h, 0, vi->depth, InputOutput, vi->visual, CWBorderPixel|CWColormap|CWEventMask, &swa);
	XMapWindow(dpy, win);
	ctx = glXCreateContext(dpy, vi, 0, GL_TRUE);
	glXMakeCurrent (dpy, win, ctx);
	glClearColor (0, 0.5, 1, 1);
	glClear (GL_COLOR_BUFFER_BIT);
	return 0;
#else
	const EGLint confAttr[] = {
		EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
		EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
		EGL_RED_SIZE, 	8,
		EGL_GREEN_SIZE, 8,
		EGL_BLUE_SIZE, 	8,
		EGL_ALPHA_SIZE, 8,
		EGL_DEPTH_SIZE, 16,
		EGL_NONE
	};
	EGLint contextAttr[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
	// Size here doesn't matter, we will draw to an offscreen buffer
	EGLint surfaceAttr[] = { EGL_WIDTH, w, EGL_HEIGHT, h, EGL_NONE };
	EGLConfig config;
	EGLint majorVersion, minorVersion, numConfigs;

	display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
	if(display == EGL_NO_DISPLAY)
	{
		eprintf("No display: 0x%x\n", eglGetError());
		return -1;
	}
	if(!eglInitialize(display, &majorVersion, &minorVersion))
	{
		eprintf("eglInitialize failed: 0x%x\n", eglGetError());
		return -1;
	}
	lprintf("eglInitialize succeeded, version=%d.%d\n", majorVersion, minorVersion);
	if(!eglGetConfigs(display, NULL, 0, &numConfigs))
	{
		eprintf("eglGetConfigs failed: 0x%x\n", eglGetError());
		return -1;
	}
	if(!eglChooseConfig(display, confAttr, &config, 1, &numConfigs))
	{
		eprintf("eglChooseConfig failed: 0x%x\n", eglGetError());
		return -1;
	}
	surface = eglCreatePbufferSurface(display, config, surfaceAttr);
	if(surface == EGL_NO_SURFACE)
	{
		eprintf("eglCreatePbufferSurface failed: 0x%x\n", eglGetError());
		return -1;
	}
	context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttr);
	if(context == EGL_NO_CONTEXT)
	{
		eprintf("eglCreateContext failed: 0x%x\n", eglGetError());
		return -1;
	}
	if(!eglMakeCurrent(display, surface, surface, context))
	{
		eprintf("eglMakeCurrent failed: 0x%x\n", eglGetError());
		return -1;
	}
	lprintf("InitEGL succeeded\n");
	return 0;
#endif
}

void CloseGL()
{
#ifdef USEGLX
	ctx = glXGetCurrentContext();
	glXDestroyContext(dpy, ctx);
	XDestroyWindow(dpy, win);
	XCloseDisplay(dpy);
#else
	eglMakeCurrent(display, surface, surface, EGL_NO_CONTEXT);
	eglDestroyContext(display, context);
	eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
	eglDestroySurface(display, surface);
	eglTerminate(display);
#endif
}

GLuint esLoadShader (GLenum type, const char *shaderSrc)
{
	GLuint shader;
	GLint compiled;

	shader = glCreateShader(type);
	if(!shader)
		return 0;
	glShaderSource(shader, 1, &shaderSrc, NULL);
	glCompileShader(shader);
	glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
	if(!compiled) 
	{
		GLint infoLen = 0;
		glGetShaderiv (shader, GL_INFO_LOG_LENGTH, &infoLen);
		if(infoLen > 1)
		{
			char* infoLog = cmalloc(sizeof(char) * infoLen);
			glGetShaderInfoLog (	shader, infoLen, NULL, infoLog);
			eprintf("Error compiling %s shader:\n%s\n",
				type == GL_VERTEX_SHADER ? "vertex" : "fragment", infoLog);
			free (infoLog);
		}
		glDeleteShader(shader);
		return 0;
	}
	return shader;
}

GLuint esLoadProgram(const char *vertShaderSrc, const char *fragShaderSrc)
{
	GLuint vertexShader;
	GLuint fragmentShader;
	GLuint programObject;
	GLint linked;

	// Load the vertex/fragment shaders
	vertexShader = esLoadShader(GL_VERTEX_SHADER, vertShaderSrc);
	if(!vertexShader)
		return 0;
	fragmentShader = esLoadShader (GL_FRAGMENT_SHADER, fragShaderSrc);
	if(!fragmentShader)
	{
		glDeleteShader(vertexShader);
		return 0;
	}
	programObject = glCreateProgram();
	if(!programObject)
		return 0;
	glAttachShader(programObject, vertexShader);
	glAttachShader(programObject, fragmentShader);
	glLinkProgram(programObject);
	glGetProgramiv(programObject, GL_LINK_STATUS, &linked);
	if(!linked) 
	{
		GLint infoLen = 0;
		glGetProgramiv(programObject, GL_INFO_LOG_LENGTH, &infoLen);
		if(infoLen > 1)
		{
			char *infoLog = cmalloc(sizeof(char) * infoLen);
		
			glGetProgramInfoLog(programObject, infoLen, NULL, infoLog);
			eprintf("Error linking program:\n%s\n", infoLog);
			free (infoLog);
		}
		glDeleteProgram(programObject);
		return 0;
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	return programObject;
}

int CreateShaders(GLSTATUS *gls)
{
	if(!programs_cache[gls->fsize][gls->quadinputs])
	{
		lprintf("Creating program for %dx%d filter, quadinputs=%d\n", gls->fsize, gls->fsize, gls->quadinputs);
		char *shader = CreateShaderSource(gls);
		programs_cache[gls->fsize][gls->quadinputs] = esLoadProgram(mVertexShader, shader);
		free(shader);
	}
	gls->programObject = programs_cache[gls->fsize][gls->quadinputs];
	if(!gls->programObject)
	{
		eprintf("esLoadProgram failed\n");
		return -1;
	}
	// Get the attribute locations
	gls->positionLoc = glGetAttribLocation(gls->programObject, "aPosition");
	gls->texCoordLoc = glGetAttribLocation(gls->programObject, "aTextureCoord");

	// Get the sampler location
	gls->sInput = glGetUniformLocation(gls->programObject, "sInput");
	gls->sFilter = glGetUniformLocation(gls->programObject, "sFilter");
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	checkGlError("glClearColor");
	return 0;
}

int CreateRenderBuffer(GLSTATUS *gls, int w, int h)
{
	lprintf("Creating render buffer\n");
	glGenRenderbuffers(1, &gls->renderbuffer);
	if(checkGlError("glGenRenderbuffers"))
		return -1;
	glBindRenderbuffer(GL_RENDERBUFFER, gls->renderbuffer);
	if(checkGlError("glBindRenderbuffer"))
		return -1;
	if(gls->quadinputs)
		glRenderbufferStorage(GL_RENDERBUFFER, gls->outtype == OUT_FLOAT ? GL_RGBA32F :
			gls->outtype == OUT_INTEGER ? GL_RGBA32I : GL_RGBA8, w, h);
	else glRenderbufferStorage(GL_RENDERBUFFER, gls->outtype == OUT_FLOAT ? GL_R32F :
			gls->outtype == OUT_INTEGER ? GL_R32I : GL_R8, w, h);
	if(checkGlError("glRenderbufferStorage"))
		return -1;
	glGenFramebuffers(1, &gls->framebuffer);
	if(checkGlError("glGenFramebuffers"))
		return -1;
	glBindFramebuffer(GL_FRAMEBUFFER, gls->framebuffer);
	if(checkGlError("glBindFramebuffer"))
		return -1;
	glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, gls->renderbuffer);
	if(checkGlError("glFramebufferRenderbuffer"))
		return -1;
	return 0;
}

static double sec[4];
static int loc, biasloc;
static GLfloat vVertices[20];
	
int Draw1(GLSTATUS *gls, INPUTS *inputs, FILTERS *filters, OUTPUTS *outputs)
{
	int i;
	const GLfloat def_vVertices[] = { -1.0f,  1.0f, 0.0f,  // Position 0
							0.0f,  1.0f,        // TexCoord 0
							-1.0f, -1.0f, 0.0f,  // Position 1
							0.0f,  0.0f,        // TexCoord 1
							1.0f, -1.0f, 0.0f,  // Position 2
							1.0f,  0.0f,        // TexCoord 2
							1.0f,  1.0f, 0.0f,  // Position 3
							1.0f,  1.0f         // TexCoord 3
							};

	if(checkGlError("start"))
		return -1;
	glViewport(0, 0, gls->otw, gls->oth);
	if(checkGlError("glViewPort"))
		return -1;

	// Clear the color buffer
	glClear(GL_COLOR_BUFFER_BIT);
	if(checkGlError("glClear"))
		return -1;

	// Use the program object
	glUseProgram(gls->programObject);
	if(checkGlError("glUseProgram"))
		return -1;
	loc = glGetUniformLocation(gls->programObject, "xstep");
	if(checkGlError("glGetUniformLocation (xstep)"))
		return -1;
	if(loc >= 0)
	{
		glUniform1f(loc, 1.0/gls->tw);
		if(checkGlError("glUniform (xstep)"))
			return -1;
	}
	loc = glGetUniformLocation(gls->programObject, "ystep");
	if(checkGlError("glGetUniformLocation (ystep)"))
		return -1;
	if(loc >= 0)
	{
		glUniform1f(loc, 1.0/gls->th);
		if(checkGlError("glUniform (ymakestep)"))
			return -1;
	}
	loc = glGetUniformLocation(gls->programObject, "pstep");
	if(checkGlError("glGetUniformLocation (pstep)"))
		return -1;
	if(loc >= 0)
	{
		glUniform1f(loc, 1.0/gls->nplanes);
		if(checkGlError("glUniform (pstep)"))
			return -1;
	}
	biasloc = glGetUniformLocation(gls->programObject, "bias");
	if(checkGlError("glGetUniformLocation (bias)"))
		return -1;
	// Load the vertex position
	glVertexAttribPointer(gls->positionLoc, 3, GL_FLOAT,
		GL_FALSE, 5 * sizeof(GLfloat), vVertices);
	if(checkGlError("glVertexAttribPointer1"))
		return -1;
	// Load the texture coordinate
	// The interpolator calculates the coordinates of the center of the texels
	// Adjust the texture coordinates for the interpolator to return the real texel coordinates
	float wmax = (float)gls->otw*filters->step / gls->tw;
	float hmax = (float)gls->oth*filters->step / gls->th;
	memcpy(vVertices, def_vVertices, sizeof(def_vVertices));
	vVertices[4] = hmax;
	vVertices[5*2 + 3] = wmax;
	vVertices[5*3 + 3] = wmax;
	vVertices[5*3 + 4] = hmax;
	for(i = 0; i < 4; i++)
	{
		vVertices[5*i + 3] -= wmax*0.5/gls->otw * (1.0 - 1.0/filters->step);
		vVertices[5*i + 4] -= hmax*0.5/gls->oth * (1.0 - 1.0/filters->step);
	}
	glVertexAttribPointer(gls->texCoordLoc, 2, GL_FLOAT,
		GL_FALSE, 5 * sizeof(GLfloat), &vVertices[3]);
	if(checkGlError("glVertexAttribPointer2"))
		return -1;

	glEnableVertexAttribArray(gls->positionLoc);
	glEnableVertexAttribArray(gls->texCoordLoc);
	if(checkGlError("glEnableVertexAttribArray"))
		return -1;

	sec[0] = seconds();
	if(gls->floattexture)
	{
		if(CreateTextureFloatInput(gls, inputs))
			return -1;
	} else {
		if(CreateTextureByteInput(gls, inputs))
			return -1;
	}
	lprintf("Running convolutions\n");
	sec[1] = seconds();
	times[0] += sec[1] - sec[0];
	return 0;
}

int Draw2(GLSTATUS *gls, INPUTS *inputs, FILTERS *filters, OUTPUTS *outputs)
{
	GLushort indices[] = { 0, 1, 2, 0, 2, 3 };
	int filter;

	for(filter = 0; filter < filters->nfilters; filter++)
	{
		sec[0] = seconds();
		if(biasloc >= 0)
		{
			glUniform1f(biasloc, filters->bias ? filters->bias[filter] : 0.0);
			if(checkGlError("glUniform (bias)"))
				return -1;
		}
		if(gls->usebuffer)
		{
			if(CreateFilterFloatInputBuffer(gls, filters, filter))
				return -1;
		} else {
			if(CreateFilterFloatInput(gls, filters, filter))
				return -1;
		}
		sec[1] = seconds();
		lprintf("Drawing %d/%d\r", filter+1, filters->nfilters);
		fflush(stdout);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, indices);
		if(checkGlError("glDrawElements"))
			return -1;
		glFinish();
		if(checkGlError("glFinish"))
			return -1;
		sec[2] = seconds();
		if(gls->outtype == OUT_FLOAT || gls->outtype == OUT_INTEGER)
		{
			if(gls->quadinputs)
			{
				if(ReadFrameFloatQuadInputs(gls, outputs, filter))
					return -1;
			} else {
				if(ReadFrameFloat(gls, outputs, filter))
					return -1;
			}
		} else {
			if(gls->quadinputs)
			{
				if(ReadFrameByteQuadInputs(gls, outputs, filter))
					return -1;
			} else {
				if(ReadFrameByte(gls, outputs, filter))
					return -1;
			}
		}
		if(gls->usebuffer)
			glDeleteBuffers(1, &gls->ftId);
		else glDeleteTextures (1, &gls->ftId);
		sec[3] = seconds();
		times[0] += sec[1] - sec[0];
		times[1] += sec[2] - sec[1];
		times[2] += sec[3] - sec[2];
	}
	glDeleteTextures (1, &gls->tId);
	lprintf("%f seconds for %f Gops, %f Gops/s\n", (times[0]+times[1]+times[2]), Gops,
		Gops/(times[0]+times[1]+times[2]));
	if(printoutput == 1)
		PrintNonZero(outputs, "GPU");
	return 0;
}

int Draw(GLSTATUS *gls, INPUTS *inputs, FILTERS *filters, OUTPUTS *outputs)
{
	if(Draw1(gls, inputs, filters, outputs))
		return -1;
	return Draw2(gls, inputs, filters, outputs);
}

///
// Cleanup
//
void ClearGLS(GLSTATUS *gls)
{
	if(gls->programObject)
	{
		//glDeleteProgram(gls->programObject);
		gls->programObject= 0;
	}
	if(gls->framebuffer)
	{
		glDeleteFramebuffers(1, &gls->framebuffer);
		gls->framebuffer = 0;
	}
	if(gls->renderbuffer)
	{
		glDeleteRenderbuffers(1, &gls->renderbuffer);
		gls->renderbuffer = 0;
	}
}

void FreeData(INPUTS *in, FILTERS *f, OUTPUTS *out)
{	
	if(in->data)
	{
		free(in->data);
		in->data = 0;
	}
	if(f->data)
	{
		free(f->data);
		f->data = 0;
	}
	if(out->data)
	{
		free(out->data);
		out->data = 0;
	}
}

void ShutDown()
{
	ClearGLS(&gls);
	FreeData(&inputs, &filters, &outputs);
	CloseGL();
}

void CalcTiling(GLSTATUS *gls, INPUTS *i, FILTERS *f, OUTPUTS *o)
{
	// Calculate the square-most tiling
	if(i->nimages % 4 == 0)
	{
		gls->quadinputs = 1;
		gls->mh = (int)sqrt(i->nimages * 0.25 + 0.1);
		gls->mh = pow(2.0, (int)(log(gls->mh)/log(2) + 0.1));
		gls->mw = (int)(i->nimages * 0.25 / gls->mh + 0.1);
		gls->nplanes = i->nplanes;
		if(gls->useinterpolator)
			lprintf("quad inputs algorithm using the interpolator\n");
		else if(gls->usebuffer)
			lprintf("quad inputs algorithm using uniform buffer for the filter\n");
		else lprintf("quad inputs algorithm\n");
	} else {
		gls->mh = (int)sqrt(i->nimages + 0.1);
		gls->mh = pow(2.0, (int)(log(gls->mh)/log(2) + 0.1));
		gls->mw = (int)(i->nimages / gls->mh + 0.1);
		gls->nplanes = (i->nplanes + 3) / 4;
	}
	gls->w = (i->width + 2*i->padding + f->step - 1) / f->step * f->step;
	gls->h = (i->height + 2*i->padding + f->step - 1) / f->step * f->step;
	gls->tw = gls->w * gls->mw;
	gls->th = gls->h * gls->mh;
	gls->ow = gls->w / f->step;
	gls->oh = gls->h / f->step;
	gls->otw = gls->ow * gls->mw;
	gls->oth = gls->oh * gls->mh;
	gls->ostride = gls->otw * 4;	// Only used for quadinputs
	gls->stride = gls->tw * 4;
	gls->pstride = gls->stride * gls->th;
	o->nimages = i->nimages;
	o->nfilters = f->nfilters;
	o->height = (i->height + 2*i->padding - f->size) / f->step + 1;
	o->width = (i->width + 2*i->padding - f->size) / f->step + 1;
	gls->fsize = f->size;
	lprintf("Input: %dx%dx%dx%d(+%d), filter: %dx%dx%dx%d(x%d), output: %dx%dx%dx%d\n",
		i->nimages, i->nplanes, i->height, i->width, i->padding,
		f->nfilters, f->nplanes, f->size, f->size, f->step,
		o->nimages, o->nfilters, o->height, o->width);
	lprintf("%dx%d tiling, texture %dx%d, render buffer %dx%d\n", gls->mw, gls->mh, gls->tw, gls->th, gls->otw, gls->oth);
	lprintf("input tile %dx%d, output tile %dx%d\n", gls->w, gls->h, gls->ow, gls->oh);
	lprintf("texture depth=%d, stride=%d, pstride=%d\n", gls->nplanes, gls->stride, gls->pstride);
}

int Init()
{
	int n;

	CalcTiling(&gls, &inputs, &filters, &outputs);
	if(InitGL(gls.otw, gls.oth))
		return -1;
	lprintf("OpenGL version: %s\n", glGetString(GL_VERSION));
	lprintf("OpenGL vendor: %s\n", glGetString(GL_VENDOR));
	glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &n);
	checkGlError("err");
	printf("MAX_3D_TEXTURE_SIZE=%d\n", n);
	glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &n);
	checkGlError("err");
	printf("MAX_ARRAY_TEXTURE_LAYERS=%d\n", n);
	lprintf("Creating data\n");
	filters.nplanes = inputs.nplanes;
	if(simpleinput)
		SimpleInputs(&inputs);
	else RandomInputs(&inputs);
	if(simplefilter)
		SimpleFilters(&filters);
	else RandomFilters(&filters);
	if(!nocheck)
	{
		lprintf("Convolving with CPU\n");
		outputs_cpu = outputs;
		Convolve(&inputs, &filters, &outputs_cpu);
		if(printoutput == 1)
			PrintNonZero(&outputs_cpu, "CPU");
	}
	lprintf("Creating shaders\n");
	if(CreateShaders(&gls))
		return -1;
	if(gls.outtype)
	{
		if(CreateRenderBuffer(&gls, gls.otw, gls.oth))
			return -1;
	}
	Gops = 1e-9 * filters.nfilters * inputs.nimages * inputs.nplanes * outputs.width * outputs.height *
		filters.size * filters.size * 2;
	AllocOutput(&inputs, &filters, &outputs);
	return 0;
}

void Run()
{
	int i;
	const char *typenames[] = {"BYTE EGL", "BYTE", "FLOAT (FAKED)", "FLOAT"};

	lprintf("Running filter size=%d, images=%d, planes=%d, filters=%d, size=%dx%d, outtype=%d\n", filters.size, inputs.nimages,
		inputs.nplanes, filters.nfilters, inputs.width, inputs.height, gls.outtype);
	times[0] = times[1] = times[2] = 0;
	for(i = 0; i < loops; i++)
		if(Draw(&gls, &inputs, &filters, &outputs))
			break;
	if(i == loops)
	{
		sprintf(sresult, "%f\t%f\t%f\t%f\t%f\t", times[0], times[1], times[2],
			(double)loops * Gops / (times[0]+times[1]+times[2]),
			(double)loops * Gops / times[1]);
		if(!nocheck)
			MeasureError(&outputs, &outputs_cpu, filters.size);
		else strcat(sresult, "\n");
		lprintf("Times Gops/s Error: %s", sresult);
		if(fpbatch && i == loops)
			fprintf(fpbatch, "%d\t%d\t%d\t%dx%d\t%s\t%s", filters.size, inputs.nimages, inputs.nplanes, inputs.width, inputs.height, typenames[gls.outtype], sresult);
	}
}

void RunBatch(const char *path)
{
	fpbatch = fopen(path, "w");
	if(!fpbatch)
	{
		eprintf("Error opening output file %s\n", path);
		return;
	}
	fprintf(fpbatch, "Zero-copy\tShader\tSize\tTypes\tWrite data\tCalc\tRead data\tSpeed (Gops/s)\tCalc-only speed\tError\n");
	if(!Init())
		Run();
	ShutDown();
	fclose(fpbatch);
	fpbatch = 0;
}

int main (int argc, char *argv[])
{
	int i, batch = 0;

	// Some defaults
	logging = 1;
	inputs.width = inputs.height = 128;
	inputs.nimages = 1;
	filters.nplanes = inputs.nplanes = 4;
	inputs.padding = 0;
	filters.step = 1;
	filters.size = 3;
	filters.nfilters = 1;
	gls.outtype = OUT_BYTE;
	if(argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")))
	{
		printf("Syntax: glconv [-fsize size] [-size size] [-planes n] [-images n] [-filters n]\n");
		printf("               [-is] [-if] [-o<b/i/f>] [-p<o/O>] [-loops loops] [-nocheck]\n");
		printf("               [-highp] [-buff] [-intp] [-batchmode]\n");
		printf("   -if = float input\n");
		printf("   -o<b/i/f> = byte/int/float output\n");
		printf("   -po n = print output (0 = only!=0, 1 = all, 2 = raw textures)\n");
		printf("   -is = simple (non random) input\n");
		printf("   -highp = high precision\n");
		printf("   -nocheck = don't run convolutions on CPU and don't check the results\n");
		printf("   -buff = use uniform buffers instead of textures for filters\n");
		printf("   -intp = use the OpenGL interpolator\n");
		printf("   -batch = run multiple predefined tests in batch\n");
		printf("Default: -fsize 3 -size 128 -planes 4 -images 1 -filters 1 -ob\n");
		return -1;
	}
	for(i = 1; i < argc; i++)
	{
		if(!strcmp(argv[i], "-size") && i+1 < argc)
			inputs.width = inputs.height = atoi(argv[++i]);
		else if(!strcmp(argv[i], "-step") && i+1 < argc)
			filters.step = atoi(argv[++i]);
		else if(!strcmp(argv[i], "-images") && i+1 < argc)
			inputs.nimages = atoi(argv[++i]);
		else if(!strcmp(argv[i], "-filters") && i+1 < argc)
			filters.nfilters = atoi(argv[++i]);
		else if(!strcmp(argv[i], "-planes") && i+1 < argc)
			filters.nplanes = inputs.nplanes = atoi(argv[++i]);
		else if(!strcmp(argv[i], "-loops") && i+1 < argc)
			loops = atoi(argv[++i]);
		else if(!strcmp(argv[i], "-buff"))
			gls.usebuffer = 1;
		else if(!strcmp(argv[i], "-intp"))
			gls.usebuffer = gls.useinterpolator = 1;
		else if(!strcmp(argv[i], "-is"))
			simpleinput = simplefilter = 1;
		else if(!strcmp(argv[i], "-if"))
			gls.floattexture = 1;
		else if(!strcmp(argv[i], "-ob"))
			gls.outtype = OUT_BYTE;
		else if(!strcmp(argv[i], "-oi"))
			gls.outtype = OUT_INTEGER;
		else if(!strcmp(argv[i], "-of"))
			gls.outtype = OUT_FLOAT;
		else if(!strcmp(argv[i], "-po") && i+1 < argc)
			printoutput = atoi(argv[++i]) + 1;
		else if(!strcmp(argv[i], "-fsize") && i+1 < argc)
			filters.size = atoi(argv[++i]);
		else if(!strcmp(argv[i], "-batch"))
			batch = 1;
		else if(!strcmp(argv[i], "-nocheck"))
			nocheck = 1;
		else if(!strcmp(argv[i], "-highp"))
			gls.highp = 1;
	}
	if(batch)
		RunBatch("glconv_out.txt");
	else {
		if(Init())
			return -1;
		Run();
		ShutDown();
	}
	return 0;
}

#ifndef NOLIB
// This is used to keep existing initializations if possible
static int glinit, curfsize, curtw, curth;
static pthread_t asyncconv_tid;
static INPUTS i_;
static FILTERS f_;
static OUTPUTS o_;

void *asyncconv_thread(void *dummy)
{
	if(Draw(&gls, &i_, &f_, &o_))
		return (void *)-1;
	return 0;
}

static int conv(lua_State *L, int async)
{
	setlocale(LC_NUMERIC, "C");
	if(!luaT_typename(L, 1) ||
		!luaT_typename(L, 2) ||
		!luaT_typename(L, 3) ||	
		strcmp(luaT_typename(L, 1), "torch.FloatTensor") ||
		strcmp(luaT_typename(L, 2), "torch.FloatTensor") ||
		strcmp(luaT_typename(L, 3), "torch.FloatTensor"))
		luaL_error(L, "<glconv>: wrong parameter types");
	THFloatTensor *t_i = luaT_toudata(L, 1, luaT_typenameid(L, "torch.FloatTensor"));
	THFloatTensor *t_f = luaT_toudata(L, 2, luaT_typenameid(L, "torch.FloatTensor"));
	THFloatTensor *t_o = luaT_toudata(L, 3, luaT_typenameid(L, "torch.FloatTensor"));

	if(t_i->nDimension < 3 || t_i->nDimension > 4 || t_f->nDimension != 4)
		luaL_error(L, "<glconv>: wrong tensor dimensions");
	if(t_f->size[2] != t_f->size[3])
		luaL_error(L, "<glconv>: only square filters supported");
	
	INPUTS *i = &i_;
	FILTERS *f = &f_;
	OUTPUTS *o = &o_;
	GLSTATUS *g = &gls;
	
	if(t_i->nDimension == 3)
	{
		if(t_i->size[0] != t_f->size[1]) // nInputPlanes
			luaL_error(L, "<glconv>: wrong tensor sizes");
		i->nimages = 1;
		i->nplanes = t_i->size[0];
		i->height = t_i->size[1];
		i->width = t_i->size[2];
	} else //if(t_i->nDimension == 4)
	{
		if(t_i->size[1] != t_f->size[1]) // nInputPlanes
			luaL_error(L, "<glconv>: wrong tensor sizes");
		i->nimages = t_i->size[0];
		i->nplanes = t_i->size[1];
		i->height = t_i->size[2];
		i->width = t_i->size[3];
	}
	if(luaT_typename(L, 4) && !strcmp(luaT_typename(L, 4), "torch.FloatTensor"))
	{
		THFloatTensor *t_b = luaT_toudata(L, 4, luaT_typenameid(L, "torch.FloatTensor"));
		if(t_b->nDimension != 1)
			luaL_error(L, "<glconv>: wrong tensor dimensions");
			
		if(t_b->size[0] != t_f->size[0])
			luaL_error(L, "<glconv>: wrong tensor sizes");
		f->bias = THFloatTensor_data(t_b);
	}
	i->padding = lua_tointeger(L, 5);
	f->step = lua_tointeger(L, 6);
	if(!f->step)
		f->step = 1;
	f->nfilters = t_f->size[0];
	f->nplanes = t_f->size[1];
	f->size = t_f->size[2];
	gls.usebuffer = gls.useinterpolator = 0;
	if(i->nimages % 4 == 0)
	{
		gls.usebuffer = 1;
		if(useinterpolator && f->step == 1)
			gls.useinterpolator = 1;
	}
	CalcTiling(&gls, i, f, o);
	if(t_i->nDimension == 3)
		THFloatTensor_resize3d(t_o, f->nfilters, o->height, o->width);
	else THFloatTensor_resize4d(t_o, i->nimages, f->nfilters, o->height, o->width);
	i->data = THFloatTensor_data(t_i);
	f->data = THFloatTensor_data(t_f);
	o->data = THFloatTensor_data(t_o);
	if(precision & 2)
	{
		g->outtype = OUT_INTEGER;
		g->floattexture = 1;
	} else {
		g->outtype = OUT_BYTE;
		g->floattexture = 0;
	}
	g->highp = precision & 1;
	if(!glinit)
	{
		glinit = 1;
		if(InitGL(gls.otw, gls.oth))
			luaL_error(L, "<glconv>: Error initializing EGL");
	}
	if(curfsize != f->size)
	{
		if(g->programObject)
			g->programObject= 0;
		curfsize = f->size;
		if(CreateShaders(&gls))
			luaL_error(L, "<glconv>: Error creating shaders");
	}
	if(g->tw != curtw || g->th != curth)
	{
		if(g->framebuffer)
		{
			glDeleteFramebuffers(1, &g->framebuffer);
			g->framebuffer = 0;
		}
		if(g->renderbuffer)
		{
			glDeleteRenderbuffers(1, &g->renderbuffer);
			g->renderbuffer = 0;
		}
		curtw = g->tw;
		curth = g->th;
		if(CreateRenderBuffer(&gls, gls.otw, gls.oth))
			luaL_error(L, "<glconv>: Error creating render buffer");
	}
	Gops = 1e-9 * f->nfilters * i->nimages * i->nplanes * o->width * o->height * f->size * f->size * 2;
	times[0] = times[1] = times[2] = 0;
	if(Draw1(&gls, i, f, o))
		luaL_error(L, "<glconv>: Error drawing");
	if(async)
	{
		pthread_create(&asyncconv_tid, 0, asyncconv_thread, 0);
	} else if(Draw2(&gls, i, f, o))
		luaL_error(L, "<glconv>: Error drawing");
	return 0;
}

static int setlogging(lua_State *L)
{
	logging = lua_tointeger(L, 1);
	return 0;
}

static int setprecision(lua_State *L)
{
	precision = lua_tointeger(L, 1);
	return 0;
}

static int useintp(lua_State *L)
{
	useinterpolator = lua_tointeger(L, 1);
	return 0;
}

static int clear(lua_State *L)
{
	int i, j;
	
	for(i = 0; i < 20; i++)
		for(j = 0; j < 2; j++)
			if(programs_cache[i][j])
			{
				glDeleteProgram(programs_cache[i][j]);
				programs_cache[i][j] = 0;
			}
	if(gls.framebuffer)
	{
		glDeleteFramebuffers(1, &gls.framebuffer);
		gls.framebuffer = 0;
	}
	if(gls.renderbuffer)
	{
		glDeleteRenderbuffers(1, &gls.renderbuffer);
		gls.renderbuffer = 0;
	}
	CloseGL();
	glinit = 0;
	curfsize = 0;
	curtw = curth = 0;
	return 0;
}

int syncconv(lua_State *L)
{
	return conv(L, 0);
}

int asyncconv(lua_State *L)
{
	return conv(L, 1);
}

int waitconv(lua_State *L)
{
	if(asyncconv_tid)
	{
		void *status;
		pthread_join(asyncconv_tid, &status);
		asyncconv_tid = 0;
		if(status)
			luaL_error(L, "<glconv>: Error drawing\n");
	} else luaL_error(L, "<glconv>: call asyncconv first");
	return 0;
}

static const struct luaL_reg libglconv[] = {
	{"logging", setlogging},
	{"clear", clear},
	{"precision", setprecision},
	{"useintp", useintp},
	{"conv", syncconv},
	{"asyncconv", asyncconv},
	{"waitconv", waitconv},
	{NULL, NULL}
};

// Initialize the library
int luaopen_libglconv(lua_State * L)
{
	luaL_register(L, "libglconv", libglconv);
	return 1;
}
#endif
