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

#include "tcv_barycentric.h"
#include "tcv.h"
#include "tcv_simd.h"
#include <iostream>

//
// SSE4 macros, as they are subsequently employed by several variants
//
#define BARY_INITIALIZE_128 \
const float eps = 1.0e-6f; \
const cv::Vec2f a = vec[0]-v; \
const cv::Vec2f b = vec[1]-v; \
const float tmp = std::fabs(detr( a,b )/2.0f); \
const float * pv = (float*)vec.data(); \
const __m128 ptx = _mm_set1_ps( v[0] ); \
const __m128 pty = _mm_set1_ps( v[1] ); \
const __m128 vEps = _mm_set1_ps( eps ); \
__m128 curx_ = _mm_set_ps( b[0], a[0], 0,0 ); \
__m128 cury_ = _mm_set_ps( b[1], a[1], 0,0 ); \
__m128 curR_ = _mm_set_ps( cv::norm( b ), cv::norm( a ), 0, 0 ); \
__m128 prevA_ = _mm_set_ps( tmp<=eps ? 0.0f : 1.0f/tmp, 0, 0, 0 ); \
__m128 prevD_ = _mm_set_ps( a.dot( b ), 0, 0, 0 ); \
__m128 vMask = _mm_castsi128_ps( _mm_set1_epi32( 0 ) ); \
__m128 vMask2 = _mm_castsi128_ps( _mm_set1_epi32( 0 ) ); \
__m128i vIndices = _mm_setr_epi32( 1,2,3,4 );

#define BARY_LOOP_128 \
__m128 nextx = _mm_loadu_ps( pv + 2*i+2 ); \
__m128 nexty = _mm_loadu_ps( pv + 2*i+6 ); \
tt_shuffle_xy( nextx, nexty ); \
nextx = _mm_sub_ps( nextx, ptx ); \
nexty = _mm_sub_ps( nexty, pty ); \
const __m128 curx = tt_compose_offset1( curx_, nextx ); \
const __m128 cury = tt_compose_offset1( cury_, nexty ); \
const __m128 vA = tt_abs_ps( _mm_mul_ps( tt_detr( curx, cury, nextx, nexty ), _mm_set1_ps( 0.5f ) ) ); \
const __m128 A_zero = _mm_cmp_ps( vA, vEps, _CMP_LE_OQ ); \
const __m128 curA = _mm_blendv_ps( _mm_div_ps( _mm_set1_ps( 1.0f ), _mm_blendv_ps( vA, _mm_set1_ps( 1.0f ), A_zero ) ), _mm_set1_ps( 0.0f ), A_zero ); \
const __m128 curD = tt_dot(curx,cury,nextx,nexty); \
const __m128 nextR = tt_norm( nextx, nexty ); \
const __m128 curR = tt_compose_offset1( curR_, nextR ); \
const __m128 R_zero =  _mm_cmp_ps( curR, vEps, _CMP_LT_OQ ); \
vMask = _mm_blendv_ps( vMask, _mm_castsi128_ps(vIndices), R_zero ); \
const __m128 invR = _mm_div_ps( _mm_set1_ps( 1.0f ), _mm_blendv_ps( curR, _mm_set1_ps( 1.0f ), R_zero ) ); \
const __m128 w1 = _mm_mul_ps( curA, _mm_fnmadd_ps( curD, invR, nextR ) ); \
const __m128 w2 = _mm_mul_ps( tt_compose_offset1( prevA_, curA ), _mm_fnmadd_ps( tt_compose_offset1( prevD_, curD ), invR, tt_compose_offset2( curR_, nextR ) ) ); \
vMask2 = _mm_blendv_ps( vMask2,_mm_castsi128_ps(vIndices), _mm_and_ps( _mm_cmp_ps( curD, _mm_set1_ps( 0.0f ), _CMP_LE_OQ ), A_zero ) ); \
vIndices =_mm_add_epi32( vIndices, _mm_set1_epi32( 4 ) ); \
curx_ = nextx; \
cury_ = nexty; \
curR_ = nextR; \
prevD_ = curD; \
prevA_ = curA;


cv::Vec3f calculate_weight_evaluate128(
	const std::vector< std::vector<cv::Vec3f> > & f_, 
	const std::vector< std::vector<cv::Vec2f> > & vec_, 
	const cv::Vec2f & v,
    const std::vector<size_t> & n_
	)
{
	if ( f_.size()!=vec_.size() || f_.size()!=n_.size() )
		CV_Error(cv::Error::StsError,"error");
	for ( size_t i=0;i<f_.size(); ++i )
	{
		size_t n = n_[i];
		size_t siz = f_[i].size();
		if ( siz < n+2 || siz !=vec_[i].size() )
			CV_Error(cv::Error::StsError,"error");
	}

	__m128 accumVal1 = _mm_set1_ps(0.0);
	__m128 accumVal2 = _mm_set1_ps(0.0);
	__m128 accumVal3 = _mm_set1_ps(0.0);
	__m128 accumWeight = _mm_set1_ps(0.0);

	for ( size_t j=0;j<f_.size(); ++j )
	{
		const std::vector< cv::Vec3f > & f = f_[j];
		const std::vector<cv::Vec2f> & vec = vec_[j];
		const size_t n = n_[j];
		if ( n < 2 )
			continue;
		int extra = ((n +3) & ~3) - n;
		const __m128 extramask = _mm_castsi128_ps(
			extra ==0 ? _mm_set1_epi32(0xffffffff) : (
				extra==3 ? _mm_setr_epi32(0xffffffff,0,0,0) :
					(extra==2 ? _mm_setr_epi32(0xffffffff,0xffffffff,0,0) : 
						_mm_setr_epi32(0xffffffff,0xffffffff,0xffffffff,0) ) ) );

		//std::cout << "n=" << n << "\n";
		//std::cout << "size= " << f.size() << "\n";
		//std::cout << "extra= " << extra << "\n";	

		BARY_INITIALIZE_128

		const float * __restrict fPtr = (float*)f.data();
		const int imax = n-4;
		int i = 1;
		for ( ; i <= imax; i+=4 )
		{
			//std::cout << "i=" << i << " next last = " << (i+3+1) << "\n";
			BARY_LOOP_128

			__m128 val1 = _mm_loadu_ps( fPtr+i*3 );
			__m128 val2 = _mm_loadu_ps( fPtr+i*3+4 );
			__m128 val3 = _mm_loadu_ps( fPtr+i*3+8 );
			tt_shuffle_xyz( val1, val2, val3 );
			__m128 w = _mm_add_ps( w1, w2 );
			accumVal1 = _mm_fmadd_ps( w, val1, accumVal1 );
			accumVal2 = _mm_fmadd_ps( w, val2, accumVal2 );
			accumVal3 = _mm_fmadd_ps( w, val3, accumVal3 );
			accumWeight = _mm_add_ps( w, accumWeight );
		}
		//std::cout << std::endl;
		// last round
		BARY_LOOP_128
		__m128 val1 = _mm_loadu_ps( fPtr+i*3 );
		__m128 val2 = _mm_loadu_ps( fPtr+i*3+4 );
		__m128 val3 = _mm_loadu_ps( fPtr+i*3+8 );
		tt_shuffle_xyz( val1, val2, val3 );
		__m128 w = _mm_and_ps( _mm_add_ps( w1, w2 ), extramask );
		accumVal1 = _mm_fmadd_ps( w, val1, accumVal1 );
		accumVal2 = _mm_fmadd_ps( w, val2, accumVal2 );
		accumVal3 = _mm_fmadd_ps( w, val3, accumVal3 );
		accumWeight = _mm_add_ps( w, accumWeight );

		__m128i mask = _mm_set1_epi32( n+1 );
		__m128i ivmask =  _mm_castps_si128(vMask);
		__m128i ivmask2 =  _mm_castps_si128(vMask2);
		
		int k;
		if ( (k=tt_reduce_max_epi32( _mm_and_si128( ivmask, _mm_cmplt_epi32( ivmask, mask ) ) )) > 0 )
		{
			return f[k-1];
		}
		else if ( (k = tt_reduce_max_epi32( _mm_and_si128( ivmask2, _mm_cmplt_epi32( ivmask2, mask ) ) )) > 0 )
		{
			float rplus = cv::norm( vec[k+1]-v );
			float r = cv::norm( vec[k]-v ); 
			cv::Vec3f fcur = f[k];
			cv::Vec3f fnext = k==n ? f[0] : f[k+1];
			return ( rplus  * fcur + r * fnext )/( r + rplus );
		}
	}
	//CV_Error(cv::Error::StsError,"error");
	cv::Vec3f retval( tt_reduce_sum_ps( accumVal1 ), tt_reduce_sum_ps( accumVal2 ), tt_reduce_sum_ps( accumVal3 ) );
	float W = tt_reduce_sum_ps( accumWeight );
	retval *= (1.0/W);
	return retval;
}

cv::Vec2f calculate_weight_evaluate128(
	const std::vector< std::vector<cv::Vec2f> > & f_, 
	const std::vector< std::vector<cv::Vec2f> > & vec_, 
	const cv::Vec2f & v,
    const std::vector<size_t> & n_
	)
{
	if ( f_.size()!=vec_.size() || f_.size()!=n_.size() )
		CV_Error(cv::Error::StsError,"error");
	for ( size_t i=0;i<f_.size(); ++i )
	{
		size_t n = n_[i];
		size_t siz = f_[i].size();
		if ( siz < n+2 || siz !=vec_[i].size() )
			CV_Error(cv::Error::StsError,"error");
	}

	__m128 accumVal1 = _mm_set1_ps(0.0);
	__m128 accumVal2 = _mm_set1_ps(0.0);
	__m128 accumWeight = _mm_set1_ps(0.0);

	for ( size_t j=0;j<f_.size(); ++j )
	{
		const std::vector< cv::Vec2f > & f = f_[j];
		const std::vector<cv::Vec2f> & vec = vec_[j];
		const size_t n = n_[j];
		if ( n < 2 )
			continue;
		int extra = ((n +3) & ~3) - n;
		const __m128 extramask = _mm_castsi128_ps(
			extra ==0 ? _mm_set1_epi32(0xffffffff) : (
				extra==3 ? _mm_setr_epi32(0xffffffff,0,0,0) :
					(extra==2 ? _mm_setr_epi32(0xffffffff,0xffffffff,0,0) : 
						_mm_setr_epi32(0xffffffff,0xffffffff,0xffffffff,0) ) ) );

		BARY_INITIALIZE_128

		const float * __restrict fPtr = (float*)f.data();
		const int imax = n-4;
		int i = 1;
		for ( ; i <= imax; i+=4 )
		{
			BARY_LOOP_128

			__m128 val1 = _mm_loadu_ps( fPtr+i*2 );
			__m128 val2 = _mm_loadu_ps( fPtr+i*2+4 );
			tt_shuffle_xy( val1, val2 );
			__m128 w = _mm_add_ps( w1, w2 );
			accumVal1 = _mm_fmadd_ps( w, val1, accumVal1 );
			accumVal2 = _mm_fmadd_ps( w, val2, accumVal2 );
			accumWeight = _mm_add_ps( w, accumWeight );
		}
		// last round
		BARY_LOOP_128
		__m128 val1 = _mm_loadu_ps( fPtr+i*2 );
		__m128 val2 = _mm_loadu_ps( fPtr+i*2+4 );
		tt_shuffle_xy( val1, val2 );
		__m128 w = _mm_and_ps( _mm_add_ps( w1, w2 ), extramask );
		accumVal1 = _mm_fmadd_ps( w, val1, accumVal1 );
		accumVal2 = _mm_fmadd_ps( w, val2, accumVal2 );
		accumWeight = _mm_add_ps( w, accumWeight );

		__m128i mask = _mm_set1_epi32( n+2 );
		__m128i ivmask =  _mm_castps_si128(vMask);
		__m128i ivmask2 =  _mm_castps_si128(vMask2);
		
		int k;
		if ( (k=tt_reduce_max_epi32( _mm_and_si128( ivmask, _mm_cmplt_epi32( ivmask, mask ) ) )) > 0 )
		{
			return f[k];
		}
		else if ( (k = tt_reduce_max_epi32( _mm_and_si128( ivmask2, _mm_cmplt_epi32( ivmask2, mask ) ) )) > 0 )
		{
			float rplus = cv::norm( vec[k+1]-v );
			float r = cv::norm( vec[k]-v ); 
			cv::Vec2f fcur = f[k];
			cv::Vec2f fnext = k==n ? f[0] : f[k+1];
			return ( rplus  * fcur + r * fnext )/( r + rplus );
		}
	}
	cv::Vec2f retval( tt_reduce_sum_ps( accumVal1 ), tt_reduce_sum_ps( accumVal2 ) );
	float W = tt_reduce_sum_ps( accumWeight );
	retval *= (1.0f/W);
	return retval;
}

float calculate_weight_evaluate128(
	const std::vector< std::vector<float> > & f_, 
	const std::vector< std::vector<cv::Vec2f> > & vec_, 
	const cv::Vec2f & v,
    const std::vector<size_t> & n_
	)
{
	if ( f_.size()!=vec_.size() || f_.size()!=n_.size() )
		CV_Error(cv::Error::StsError,"error");
	for ( size_t i=0;i<f_.size(); ++i )
	{
		size_t n = n_[i];
		size_t siz = f_[i].size();
		if ( siz < n+2 || siz !=vec_[i].size() )
			CV_Error(cv::Error::StsError,"error");
	}

	__m128 accumVal1 = _mm_set1_ps(0.0);
	__m128 accumWeight = _mm_set1_ps(0.0);

	for ( size_t j=0;j<f_.size(); ++j )
	{
		const std::vector< float > & f = f_[j];
		const std::vector<cv::Vec2f> & vec = vec_[j];
		const size_t n = n_[j];
		if ( n < 2 )
			continue;
		int extra = ((n +3) & ~3) - n;
		const __m128 extramask = _mm_castsi128_ps(
			extra ==0 ? _mm_set1_epi32(0xffffffff) : (
				extra==3 ? _mm_setr_epi32(0xffffffff,0,0,0) :
					(extra==2 ? _mm_setr_epi32(0xffffffff,0xffffffff,0,0) : 
						_mm_setr_epi32(0xffffffff,0xffffffff,0xffffffff,0) ) ) );

		BARY_INITIALIZE_128

		const float * __restrict fPtr = (float*)f.data();
		const int imax = n-4;
		int i = 1;
		for ( ; i <= imax; i+=4 )
		{
			BARY_LOOP_128

			__m128 val1 = _mm_loadu_ps( fPtr+i );
			__m128 w = _mm_add_ps( w1, w2 );
			accumVal1 = _mm_fmadd_ps( w, val1, accumVal1 );
			accumWeight = _mm_add_ps( w, accumWeight );
		}
		// last round
		BARY_LOOP_128
		__m128 val1 = _mm_loadu_ps( fPtr+i );	
		__m128 w = _mm_and_ps( _mm_add_ps( w1, w2 ), extramask );
		accumVal1 = _mm_fmadd_ps( w, val1, accumVal1 );
		accumWeight = _mm_add_ps( w, accumWeight );

		__m128i mask = _mm_set1_epi32( n+2 );
		__m128i ivmask =  _mm_castps_si128(vMask);
		__m128i ivmask2 =  _mm_castps_si128(vMask2);
		
		int k;
		if ( (k=tt_reduce_max_epi32( _mm_and_si128( ivmask, _mm_cmplt_epi32( ivmask, mask ) ) )) > 0 )
		{
			return f[k];
		}
		else if ( (k = tt_reduce_max_epi32( _mm_and_si128( ivmask2, _mm_cmplt_epi32( ivmask2, mask ) ) )) > 0 )
		{
			float rplus = cv::norm( vec[k+1]-v );
			float r = cv::norm( vec[k]-v ); 
			float fcur = f[k];
			float fnext = k==n ? f[0] : f[k+1];
			return ( rplus  * fcur + r * fnext )/( r + rplus );
		}
	}
	float retval = tt_reduce_sum_ps( accumVal1 );
	float W = tt_reduce_sum_ps( accumWeight );
	retval *= (1.0f/W);
	return retval;
}

