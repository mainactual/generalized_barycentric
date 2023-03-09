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

#ifndef _tcv_
#define _tcv_

#include "opencv2/core.hpp"

// get pointer to the beginning of line y
#define GET_LINE_PTR( _Type, _bytes, _stride, _y )		( reinterpret_cast<_Type*>( (_bytes)+(_y)*(_stride) ) )
// get pointer to a given pixel at (x,y)
#define GET_PIXEL_PTR(_Type, _x, _y, _bytes, _stride )	(reinterpret_cast<_Type*>( (_bytes)+(_y)*(_stride) ) +(_x))
// get logical index from x, y
#define GET_INDEX( _x, _y, _width )						((_y)*(_width)+(_x))
// get (x,y) from logical index
#define GET_LOGICAL( _x, _y, _idx, _width )				(_y)=(_idx)/(_width); (_x)=(_idx)%(_width)
// assert a logical index is within the extent
#define VALID_LOGICAL( _x, _y, _width, _height )		((_x)>=0 && (_x)<(_width) && (_y)>=0 && (_y)<(_height))

// legacy macros
#define COORD2LINEAR	GET_INDEX
#define LINEAR2COORD	GET_LOGICAL	

namespace tcv {
template<class _T> int get_cv_type()
{
	if ( typeid(_T)==typeid(unsigned char) )
		return CV_8U;
	else if ( typeid(_T)==typeid(short) )
		return CV_16S;
	else if ( typeid(_T)==typeid(unsigned short) )
		return CV_16U;
	else if ( typeid(_T)==typeid(int) )
		return CV_32S;
	else if ( typeid(_T)==typeid(float) )
		return CV_32F;
	else if ( typeid(_T)==typeid(double) )
		return CV_64F;
	else if ( typeid(_T)==typeid(cv::Vec3b) )
		return CV_8UC3;
	CV_Error(cv::Error::StsError,"not impl");
	return -1;
}
}


// determinant of a vector
template<class Precision> Precision detr(const cv::Vec<Precision,2> & p1, const cv::Vec<Precision,2> & p2)
{
    return p1[0] * p2[1] - p1[1] * p2[0];
}

// zero check with precision
template<class Precision> bool is_zero( Precision v, Precision eps )
{
	return v>=-eps && v<=eps ? true : false;
}

#endif
