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

#ifndef _tcv_simd_
#define _tcv_simd_

#include <xmmintrin.h>
#include <immintrin.h>


/** @brief Absolute value
*/
inline __m128 tt_abs_ps( const __m128 & t )
{
	return _mm_andnot_ps( _mm_set1_ps(-0.f), t );
}
/** @brief Determinant
*/
inline __m128 tt_detr( const __m128 & x1, const __m128 & y1, const __m128 & x2, const __m128 & y2 )
{
	return _mm_fmsub_ps( x1, y2 , _mm_mul_ps( y1, x2 ) );
}
/** @brief Zero check with finite precision
*/
inline __m128i is_zero( const __m128 & v, const __m128 vEps )
{
	return _mm_and_si128(
		_mm_castps_si128(_mm_cmp_ps( v, vEps, _CMP_LE_OQ )),
		_mm_castps_si128(_mm_cmp_ps( v, _mm_mul_ps( _mm_set1_ps(-1.0f), vEps ), _CMP_GE_OQ ) ) );
}
/** @brief L2-norm
*/
inline __m128 tt_norm( const __m128 & x, const __m128 & y )
{
	return _mm_sqrt_ps( _mm_fmadd_ps( x, x, _mm_mul_ps( y, y ) ) );
}
/** @brief Dot product
*/
inline __m128 tt_dot( const __m128 & x1, const __m128 & y1, const __m128 & x2, const __m128 & y2 )
{
	return _mm_fmadd_ps( x1, x2 , _mm_mul_ps( y1, y2 ) );
}
/** @brief Given a vector_{4*n}, and a vector_{4*(n+1)}, shuffles the result as vector_{4*(n+1)-1}
*/
inline __m128 tt_compose_offset1( const __m128 & tx, const __m128 & tx2 )
{
	 return _mm_castsi128_ps( 
		 _mm_or_si128( 
			 _mm_srli_si128( _mm_castps_si128(tx), 12 ), 
			 _mm_slli_si128( _mm_castps_si128(tx2), 4 ) ));
}
/** @brief Given a vector_{4*n}, and a vector_{4*(n+1)}, shuffles the result as vector_{4*(n+1)-2}
*/
inline __m128 tt_compose_offset2( const __m128 & tx, const __m128 & tx2 )
{
	 return _mm_castsi128_ps( 
		 _mm_or_si128( 
			 _mm_srli_si128( _mm_castps_si128(tx), 8 ), 
			 _mm_slli_si128( _mm_castps_si128(tx2), 8 ) ));
}
/** @brief Reduces maximum value of packed 32-bit integers
*/
inline int tt_reduce_max_epi32(const __m128i & a )
{
	__m128i val = _mm_max_epi32(a, _mm_srli_si128(a,8));
	val = _mm_max_epi32(val, _mm_srli_si128(val,4));
	return (int) _mm_cvtsi128_si32( val );
}
/** @brief Shuffles packed vector of 2D-floats into two separate channels
* 
input: x0 y0, x1 y1 - x2 y2, x3 y3
output: x0 x1, x2 x3 - y0 y1, y2, y3
*/
inline void tt_shuffle_xy( __m128 & low, __m128 & high )
{
	__m128 tx = _mm_shuffle_ps( low, high, 0 + (2<<2) + (0<<4) + (2<<6) );
	high = _mm_shuffle_ps( low, high, 1 + (3<<2) + (1<<4) + (3<<6) );
	low = tx;
}
/** @brief Reduces sum of packed 32-bit floats
*/
inline float tt_reduce_sum_ps(const __m128 & a )
{
	__m128 val = _mm_add_ps(a, _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a),8)) );
	val = _mm_add_ps(val, _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(val),4)) );
	return _mm_cvtss_f32( val );
}
/** @brief Shuffles packed vector of 3D-floats into three separate channels
*/
inline void tt_shuffle_xyz( __m128 & a, __m128 & b, __m128 & c )
{
	__m128 x = _mm_permute_ps( _mm_blend_ps( a, _mm_blend_ps( b, c, 1 + 2 ), 2+4 ), 0 + (3<<2) + (2<<4) + (1<<6) );
	__m128 y = _mm_permute_ps( _mm_blend_ps( _mm_blend_ps( a, b, 1+8 ), c, 4 ), 1 + (0<<2) + (3<<4) + (2<<6) );
	c = _mm_permute_ps( _mm_blend_ps( _mm_blend_ps( a, c, 1+8 ), b, 2 ), 2 + (1<<2) + (0<<4) + (3<<6) );
	a = x;
	b = y;
}
#endif
