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

#include "tcv.h"
#include "tcv_barycentric.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <fstream>

//
// the following functions for 68-point face landmark detection, e.g., that provided with Dlib
//
std::vector< cv::Point > select_points( const std::vector< cv::Point> & a, const std::vector< int > & indices )
{
	std::vector<cv::Point > ret( indices.size() );
	for ( size_t i=0;i<indices.size(); ++i )
	{
		int idx = indices[i];
		if ( idx<0 || idx>=a.size() )
			CV_Error(cv::Error::StsError,"error");
		ret[i] = a[ idx ];
	}
	return ret;
}
std::vector< cv::Point > left_eye( const std::vector< cv::Point > &pts )
{
	if ( pts.size()!=68 )
		CV_Error(cv::Error::StsError,"pts!=68");
	return select_points( pts, std::vector<int>{36,37,38,39,40,41} );
}
std::vector< cv::Point > right_eye( const std::vector< cv::Point > &pts )
{
	if ( pts.size()!=68 )
		CV_Error(cv::Error::StsError,"pts!=68");
	return select_points( pts, std::vector<int>{42,43,44,45,46,47} );
}
std::vector< cv::Point > face( const std::vector< cv::Point > & pts )
{
	if ( pts.size()!=68 )
		CV_Error(cv::Error::StsError,"pts!=68");	
	std::vector< cv::Point > ret( 26 );
	for ( unsigned int u=0;u<26;++u )
		ret[u]=pts[u];
	return ret;
}
std::vector< cv::Point >  mouth(const std::vector< cv::Point > &pts)
{
	if (pts.size() != 68)
		CV_Error(cv::Error::StsError, "pts!=68");
	return select_points( pts, std::vector<int>{48,49,50,51,52,53,54,55,56,57,58,59});
}
void drawface(cv::Mat & m, const std::vector< cv::Point > & pts, cv::Scalar color, int thickness, bool with_outline)
{
	if (pts.size() != 68)
		return;
	if ( with_outline )
	{
		for (size_t i = 0; i <= 15; ++i)
			cv::line(m, pts[i], pts[i + 1], color, thickness);
	}
	for (size_t i = 17; i <= 20; ++i)
		cv::line(m, pts[i], pts[i + 1], color, thickness);
	for (size_t i = 22; i <= 25; ++i)
		cv::line(m, pts[i], pts[i + 1], color, thickness);
	for (size_t i = 27; i <= 29; ++i)
		cv::line(m, pts[i], pts[i + 1], color, thickness);
	for (size_t i = 31; i <= 34; ++i)
		cv::line(m, pts[i], pts[i + 1], color, thickness);
	for (size_t i = 36; i <= 40; ++i)
		cv::line(m, pts[i], pts[i + 1], color, thickness);
	cv::line(m, pts[36], pts[41], color, thickness);
	for (size_t i = 42; i <= 46; ++i)
		cv::line(m, pts[i], pts[i + 1], color, thickness);
	cv::line(m, pts[42], pts[47], color, thickness);
	for (size_t i = 48; i <= 58; ++i)
		cv::line(m, pts[i], pts[i + 1], color, thickness);
	cv::line(m, pts[48], pts[59], color, thickness);
	for (size_t i = 60; i <= 66; ++i)
		cv::line(m, pts[i], pts[i + 1], color, thickness);
	cv::line(m, pts[60], pts[67], color, thickness);
}
std::vector<cv::Point > subtract_points( const std::vector<cv::Point> & a, const std::vector<cv::Point> & b )
{
	std::vector<cv::Point > ret( a.size() );
	if ( a.size()!=b.size() )
		CV_Error(cv::Error::StsError,"error");
	for ( size_t i=0;i<ret.size(); ++i )
		ret[i] = b[i]-a[i];
	return ret;
}
/** @brief Reads a landmark file arranged as
Points should be arranged as:
x,y
x,y
x,y
...

Number of landmarks should be 68.
*/
std::vector<cv::Point> read_points( const std::string & filename )
{
	std::vector<cv::Point> ret;
	std::ifstream file( filename );
	std::string line;
	for ( ; std::getline( file,line ); )
	{
		std::for_each( line.begin(), line.end(), [](char &c){if(c==',')c=' ';} );
		std::istringstream line(line);
		int x,y;
		line >> x >> y;
		ret.push_back( cv::Point(x,y) );
	}
	return ret;
}


/** @brief Merges a vector of images horizontally into one, decimates on request

@param in is input vector of images. the pixel type must be same across the images
@param out is the output image
@param factor is the decimation factor or 1.0 for no decimation
*/
void merge_images( const std::vector< cv::Mat > & in, cv::Mat & out, double factor )
{
	if ( in.empty() )
		return;
	if ( factor <= 0.0 || factor > 1.0 )
		CV_Error(cv::Error::StsError,"error");

	const int type = in[0].type();

	std::vector< cv::Rect > rects( in.size() );
	int w = 0;
	int h = 0;
	for ( int i=0;i<rects.size(); ++i )
	{
		cv::Rect r( w, 0, (int)(factor * in[i].cols), (int)(factor * in[i].rows) );
		if ( !r.area() )
			CV_Error(cv::Error::StsError,"!area");
		if ( in[i].type() != type )
			CV_Error(cv::Error::StsError,"wrong type");

		rects[i] = r;
		if ( (__int64)(w + r.width) >= 0x80000000 )
			CV_Error(cv::Error::StsError,"too large");
		w += r.width;
		if ( h < r.height )
			h = r.height;
	}
	if ( (__int64)w * (__int64)h > (__int64)0x80000000 )
		CV_Error(cv::Error::StsError,"Too large");

	cv::Size target(w,h);
	if ( out.size() != target || out.type() != type )
		out = cv::Mat( target, type );
	for ( int i=0;i<rects.size(); ++i )
	{
		cv::Mat tmp = out( rects[i] );
		if ( tmp.size() == in[i].size() )
			in[i].copyTo( tmp);
		else
			cv::resize( in[i], tmp, tmp.size() );
	}
}
/** @brief Extracts channels from a multichannel image
* 
@param src is the multichannel image
@param dst is a vector of channels components
*/
void extract( const cv::Mat & src, std::vector< cv::Mat > & dst )
{
	if ( src.empty() )
		CV_Error(cv::Error::StsError,"empty");

	const unsigned int npairs = src.channels();
	if ( dst.size() != npairs )
		dst.resize( npairs );

	if ( npairs == 1 )
	{
		src.copyTo( dst[0] );
		return;
	}
	std::vector< int > fromTo( 2*npairs );
	for ( unsigned int u=0; u < npairs; ++u )
	{
		fromTo[2*u  ] = u;
		fromTo[2*u+1] = u;
	}
	int newtype = CV_MAKE_TYPE( src.depth(), 1 );
	for ( unsigned int u = 0, i=0; u < npairs; ++u, i+=2 )
	{
		if ( dst[u].size() != src.size() || dst[u].type() != newtype )
			dst[u] = cv::Mat( src.size(), newtype ); 
	}
	std::vector< cv::Mat > _src;
	_src.push_back( src );
	cv::mixChannels( _src, dst, &fromTo[0], npairs );
}
/** @brief Image warp demo
* 
@param src_img is the full name of the source image
@param src_points is the full name of the 68-points landmark file for the source
@param dst_img is the full name of the destination image
@param dst_points is the full name of the 68-points landmark file for the destination
*/
void imagewarp( 
	const std::string & src_img, 
	const std::string & src_points, 
	const std::string & dst_img,
	const std::string & dst_points )
{
	bool use_face_outline = true;

	cv::Mat a = cv::imread( dst_img, cv::IMREAD_COLOR);
	cv::Mat b = cv::imread( src_img, cv::IMREAD_COLOR);
	std::vector< cv::Point > pa = read_points( dst_points );
	std::vector< cv::Point > pb = read_points( src_points );

	typedef std::vector<cv::Vec2f> Vec;
	std::vector< Vec > polygons;
	std::vector< Vec > displacements;
	std::vector< size_t > sizes;

	std::vector< cv::Point2f > target, source;

	if ( use_face_outline )
	{
		std::vector< cv::Point > p=face( pa );
		std::vector< cv::Point > q = face( pb );

		// it is cleaner to warp only the convex hull landmarks
		std::vector< int > indices;
		cv::convexHull( p, indices, true, false );
		p = select_points( p, indices );
		q = select_points( q, indices );
		
		// add into the list of polygons
		polygons.push_back( padd_vector( p ) );
		displacements.push_back( padd_vector( subtract_points(p,q) ) );
		sizes.push_back( p.size() );
	}
	for ( int i=0;i<3;++i )
	{
		// mouth
		std::vector< cv::Point > p, q;
		if ( i==2 )
		{ 
			p=mouth( pa );
			q=mouth( pb );
		}else if ( i==1 )
		{
			p=right_eye( pa );
			q=right_eye( pb );
		}else
		{
			p=left_eye( pa );
			q=left_eye( pb );
		}
		// reverse point order, as they are nested inside the face landmarks
		std::reverse( p.begin(), p.end() );
		std::reverse( q.begin(), q.end() );

		// add also the mean points for affine transform
		cv::Scalar m = cv::mean( p );
		target.push_back( cv::Point2f( m[0],m[1] ) );
		m = cv::mean( q );
		source.push_back( cv::Point2f( m[0],m[1] ) );

		// add into the list of polygons
		polygons.push_back( padd_vector( p ) );
		displacements.push_back( padd_vector( subtract_points(p,q) ) );
		sizes.push_back( p.size() );
	}

	// fill the displacement field
	cv::Mat buffer; // for the template function
	cv::Mat displ( a.size(), CV_32FC2 );
	for ( int y=0;y<displ.rows;++y)
	{
		for ( int x=0;x<displ.cols;++x)
		{
			cv::Vec2f fval = calculate_weight_evaluate128( displacements, polygons, cv::Vec2f(x,y), sizes );
			displ.at< cv::Vec2f >(y,x) = cv::Vec2f( x,y ) + fval;
		}
	}
	// remap by barycentric coordinates
	cv::Mat c,d;
	cv::remap( b, c, displ, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_REPLICATE );
	// remap by affine transform
	cv::Mat tr = cv::getAffineTransform( target, source );
	cv::warpAffine( b, d, tr, b.size(), cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE );

	drawface( a, pa, cv::Scalar(0,0,255), 1, true );
	//drawface( b, pa, cv::Scalar(0,0,255), 1, true );
	drawface( b, pb, cv::Scalar(255,0,0), 1, true );
	drawface( c, pa, cv::Scalar(0,0,255), 1, true );
	drawface( d, pa, cv::Scalar(0,0,255), 1, true );

	cv::Mat mrg;

	merge_images( std::vector<cv::Mat>{a,b,c,d}, mrg, 0.5 );
	cv::imshow( "win", mrg );
	//cv::imwrite("D:/tmp/imagewarp.jpg", mrg );
	cv::waitKey(0);
}
void imagewarp()
{
	imagewarp(
		"D:/c2plus/bary/a.png","D:/c2plus/bary/a.txt",
		"D:/c2plus/bary/b.png","D:/c2plus/bary/b.txt" );
}
