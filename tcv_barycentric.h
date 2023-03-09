/*=========================================================================
 *
 *  Copyright mainactual
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef _tcv_barycentric_
#define _tcv_barycentric_

#include "tcv.h"
#include "opencv2/core.hpp"


/** @brief Padds a closed polygon in-place, for "calculate_weight_evaluate" functions 

Padds a closed polygon, or values associated with a closed polygon, for efficient traversal
Also, padds them suitable for 16-byte aligned SSE4. The first function is in-place. Type
should present a floating point scalar or vector, define basic math operations, and also, 
a broadcast assign initialization, s.t.,

@code
float f = 0;
cv::Vec3f f = 0; // all zeros
@endcode

@param f The source and destination vector
*/
template<class T> void padd_vector_inplace( std::vector<T> & f )
{
    if ( f.empty() )
		CV_Error(cv::Error::StsError,"error");
    const size_t n = f.size();
    const size_t targetsize = 2 + ( (n+3) & ~3 );
    f.resize( targetsize, f[0] );
    for ( size_t i = n; i>0; --i )
        f[i] = f[i-1];
    f[0]=f[n];
}
/** @brief Padds a closed polygon for "calculate_weight_evaluate" functions 

Returns padded polygon

@param f The source vector
*/
template<class T> std::vector<T> padd_vector( const std::vector<T> & f )
{
    if ( f.empty() )
		CV_Error(cv::Error::StsError,"error");
    const size_t n = f.size();
    const size_t targetsize = 2 + ( (n+3) & ~3 );
    std::vector< T > ret( targetsize, f[0] );
	for ( size_t i=1;i<=n; ++i )
		ret[i]=f[i-1];
	ret[0]=ret[n];
	return ret;
}
/** @brief Padds a closed polygon for "calculate_weight_evaluate" functions 

Returns padded polygon. 
This is convenience method, that essentially converts color, as cv::Scalar, into a cv::Vec3f

@param f The source vector
*/
inline std::vector<cv::Vec3f> padd_vector( const std::vector< cv::Scalar > & f )
{
    if ( f.empty() )
		CV_Error(cv::Error::StsError,"error");
    const size_t n = f.size();
    const size_t targetsize = 2 + ( (n+3) & ~3 );
    std::vector< cv::Vec3f > ret( targetsize, cv::Vec3f( f[0][0],f[0][1],f[0][2] ) );
	for ( size_t i=1;i<=n; ++i )
		ret[i]=cv::Vec3f( f[i-1][0],f[i-1][1],f[i-1][2] );
	ret[0]=ret[n];
	return ret;
}
/** @brief Padds a closed polygon for "calculate_weight_evaluate" functions 

Returns padded polygon. 
This is convenience method, that essentially converts cv::Point into a cv::Vec2f

@param f The source vector
*/
inline std::vector<cv::Vec2f> padd_vector( const std::vector< cv::Point > & f )
{
    if ( f.empty() )
		CV_Error(cv::Error::StsError,"error");
    const size_t n = f.size();
    const size_t targetsize = 2 + ( (n+3) & ~3 );
    std::vector< cv::Vec2f > ret( targetsize, cv::Vec2f( f[0].x, f[0].y ) );
	for ( size_t i=1;i<=n; ++i )
		ret[i]=cv::Vec2f( f[i-1].x,f[i-1].y );
	ret[0]=ret[n];
	return ret;
}

/** @brief Evaluates generalized barycentric weights at a given location

Argument type should obey rules described in "padd_vector_inplace"

@param f_ is a vector of values associated with each closed polygon in type T.
@param vec_ is a vector of polygon points in 2D 
f_ and vec_ must be padded vectors using one of "padd_vector"-functions
f_, vec_ and n_ must be of the same length
the orientation of nested polygons should change at each level

@param v is the location to be evaluated
@param buffer is a compute buffer, as this function is typically called subsequently
@param n_ is a vector of true lenghts of each polygon (excluding the padd entries)

Function is based on 

Hormann, "Barycentric coordinates for arbitrary polygons in the plane", 2014

*/
template<class T, class Precision > T calculate_weight_evaluate(
	const std::vector< std::vector<T> > & f_, 
	const std::vector< std::vector< cv::Vec<Precision,2> > > & vec_, 
	const cv::Vec<Precision,2> & v,
	cv::Mat & buffer,
    const std::vector<size_t> & n_
	)
{
	typedef cv::Vec< Precision, 2 > Vec;

	if ( f_.size()!=vec_.size() || f_.size()!=n_.size() )
		CV_Error(cv::Error::StsError,"error");
	size_t len=0;
	for ( size_t i=0;i<f_.size(); ++i )
		len = std::max( len, f_[i].size() );
	
	const Precision eps = 1.0e-6;
	int cvtype = tcv::get_cv_type<Precision>();
	if ( buffer.type()!=cvtype || buffer.rows!=3 || buffer.cols < len )
		buffer = cv::Mat( 3, len, cvtype, cv::Scalar(0.0f) );
	Precision * A = buffer.ptr<Precision>(0);
	Precision * D = buffer.ptr<Precision>(1);
	Precision * r = buffer.ptr<Precision>(2);

	T retval=0.0;
	Precision W = 0.0;

	for ( size_t i=0;i<f_.size(); ++i )
	{
		const std::vector<T> & f = f_[i];
		const std::vector< Vec > & vec = vec_[i];
		size_t n = n_[i];
		if ( n < 2 )
			continue;
		if ( vec.size() < n+2 || f.size()!=vec.size() )
			CV_Error(cv::Error::StsError,"error");

		for ( size_t i=0; i<=n;++i )
		{
			Vec cur = vec[i]-v;
			Vec next = vec[i+1]-v;
			r[i] = cv::norm( cur );
			A[i] = std::fabs( detr( cur, next )/2.0f );
			D[i] = cur.dot( next );
		}
		r[n+1] = r[1];
		r[0] = r[n];
		A[0] = A[n];
		D[0] = D[n];
		for ( size_t i=1;i<=n; ++i )
		{
			Precision w = 0.0;
			if ( r[i] < eps ) // v == v_i
				return f[i];
			if ( A[i] > eps )
			{
				w += ( r[i+1] - D[i]/r[i] )/A[i];
			}
			else if ( D[i] < 0.0f ) // v \in e_i
			{
				T fcur = f[i];
				T fnext = i==n ? f[0] : f[i+1];
				return ( r[i+1]  * fcur + r[i] * fnext )/( r[i] + r[i+1] );
			}
			if ( A[i-1] > eps )
			{
				w += ( r[i-1] - D[i-1]/r[i] )/A[i-1];
			}
			retval += w * f[i];
			W += w;
		}
	}
	retval *= (1.0f/W);
	return retval;
}


/** @brief Evaluates generalized barycentric weights at a given location

Argument type should obey rules described in "padd_vector_inplace"
Optimized SSE4 versions for several typical use cases 

@param f_ is a vector of values associated with each closed polygon in type T.
@param vec_ is a vector of polygon points in 2D 
f_ and vec_ must be padded vectors using one of "padd_vector"-functions
f_, vec_ and n_ must be of the same length

@param v is the location to be evaluated
@param n_ is a vector of true lenghts of each polygon (excluding the padd entries)
*/
cv::Vec3f calculate_weight_evaluate128(
	const std::vector< std::vector<cv::Vec3f> > & f_, 
	const std::vector< std::vector<cv::Vec2f> > & vec_, 
	const cv::Vec2f & v,
    const std::vector<size_t> & n_
	);

cv::Vec2f calculate_weight_evaluate128(
	const std::vector< std::vector<cv::Vec2f> > & f_, 
	const std::vector< std::vector<cv::Vec2f> > & vec_, 
	const cv::Vec2f & v,
    const std::vector<size_t> & n_
	);

float calculate_weight_evaluate128(
	const std::vector< std::vector<float> > & f_, 
	const std::vector< std::vector<cv::Vec2f> > & vec_, 
	const cv::Vec2f & v,
    const std::vector<size_t> & n_
	);

/** @brief Evaluates generalized barycentric weights at a given location

Argument type should obey rules described in "padd_vector_inplace"
This is scalar version of the SSE version with optional accumulator precision

@param f_ is a vector of values associated with each closed polygon in type T.
@param vec_ is a vector of polygon points in 2D 
f_ and vec_ must be padded vectors using one of "padd_vector"-functions
f_, vec_ and n_ must be of the same length

@param v is the location to be evaluated
@param n_ is a vector of true lenghts of each polygon (excluding the padd entries)
*/
template<class T, class Precision, class TAccum=T, class PrecisionAccum=Precision > T calculate_weight_evaluate(
	const std::vector< std::vector<T> > & f_, 
	const std::vector< std::vector< cv::Vec<Precision,2> > > & vec_, 
	const cv::Vec<Precision,2> & v,
    const std::vector<size_t> & n_
	)
{
	typedef cv::Vec< Precision, 2 > Vec;

	if ( f_.size()!=vec_.size() || f_.size()!=n_.size() )
		CV_Error(cv::Error::StsError,"error");
	for ( size_t i=0;i<f_.size(); ++i )
	{
		size_t n = n_[i];
		size_t siz = f_[i].size();
		if ( siz < n+2 || siz !=vec_[i].size() )
			CV_Error(cv::Error::StsError,"error");
	}
	const Precision eps = 1.0e-6;

	TAccum accumVal = 0;
	PrecisionAccum accumWeight = 0;
	for ( size_t j=0;j<f_.size(); ++j )
	{
		const std::vector< T > & f = f_[j];
		const std::vector< Vec > & vec = vec_[j];
		const size_t n = n_[j];
		if ( n < 2 )
			continue;
		
		Vec prevX = vec[0]-v;
		Vec curX = vec[1]-v;
		Precision tmp = std::fabs(detr( prevX, curX ) * 0.5 );
		Precision curR = cv::norm( curX );
		Precision prevR = cv::norm( prevX );
		Precision prevA = tmp<=eps ? 0.0 : 1.0/tmp;
		Precision prevD = prevX.dot( curX );

		int i = 1;
		for ( ; i <=n; i+=1 )
		{
			if ( curR < eps ) // v == v_i
				return f[i];
			
			Vec nextX = vec[i+1] - v;
			//r[i] = cv::norm( cur );
			Precision nextR = cv::norm( nextX );
			
			//A[i] = std::fabs( detr( cur, next )/2.0f );
			Precision A = std::fabs( detr( curX, nextX ) * 0.5 );
			Precision curA = A <= eps ? 0.0 : 1.0/A;
			//D[i] = cur.dot( next );
			Precision curD = curX.dot( nextX );
			
			if ( A <= eps && curD <= 0.0f ) // v \in e_i
			{
				T fcur = f[i];
				T fnext = i==n ? f[0] : f[i+1];
				return ( nextR  * fcur + curR * fnext )/( curR + nextR );
			}

			Precision invR = 1.0/curR; // tested already
			//if ( A[i] > eps ) w += ( r[i+1] - D[i]/r[i] )/A[i];
			Precision w1 = curA * (nextR - curD * invR );
			//if ( A[i-1] > eps )	w += ( r[i-1] - D[i-1]/r[i] )/A[i-1];
			Precision w2 = prevA * (prevR - prevD * invR );
			Precision w = w1+w2;
			accumVal += w * f[i];
			accumWeight += w;
			
			prevR = curR;
			prevD = curD;
			prevX = curX;

			curR = nextR;
			prevA = curA;
			curX = nextX;
		}
	}
	accumVal *= (1.0f/accumWeight);
	return T(accumVal);
}
/** @brief Evaluates barycentric weights on a triangle at a given location


@param f_ is a vector of values associated with each closed polygon in type T.
@param vec_ is a vector of polygon points in 2D 
f_ and vec_ must be same length; if the length is 3, it is a regular triangle
Otherwise, a padding by one is assumed

@param v is the location to be evaluated
@param background is the default value for point outside the triangle
*/
template<class T, class Precision > T calculate_weight_evaluate_triangle(
	const std::vector<T> & f_, 
	const std::vector< cv::Vec<Precision,2> > & vec_, 
	const cv::Vec<Precision,2> & v,
	const T & background
	)
{
	typedef cv::Vec< Precision, 2 > Vec;

	if ( f_.size()!=vec_.size()  || f_.size()< 3 )
		CV_Error(cv::Error::StsError,"error");

	Vec v0;
	Vec v1;
	Vec v2;
	if ( f_.size()==3 )
	{
		v0 = vec_[1]-vec_[0];
		v1 = vec_[2]-vec_[0];
		v2 = v - vec_[0];
	}else
	{
		// padded polygon
		v0 = vec_[2]-vec_[1];
		v1 = vec_[3]-vec_[1];
		v2 = v - vec_[1];
	}
	// these are unnecessarily calculated every single time
	Precision d00 = v0.dot( v0 );
	Precision d01 = v0.dot( v1 );
	Precision d11 = v1.dot( v1 );
	Precision denom = d00 * d11 - d01 * d01;
	if ( denom == 0 )
		return background;
	Precision idenom = 1.0/denom;
	Precision d20 = v2.dot( v0 );
	Precision d21 = v2.dot( v1 );
	Precision bcoords[3];
	if ( (bcoords[1] = (d11 * d20 - d01 * d21) * idenom)<0.0 )
		return background;
	if ( (bcoords[2] = (d00 * d21 - d01 * d20) * idenom)<0.0 )
		return background;
	if ( (bcoords[0] = 1.0 - bcoords[1] - bcoords[2])<0.0 )
		return background;
	if ( f_.size()==3 )
		return f_[0]*bcoords[0] + f_[1]*bcoords[1] + f_[2]*bcoords[2];
	return f_[1]*bcoords[0] + f_[2]*bcoords[1] + f_[3]*bcoords[2];
}

#endif
