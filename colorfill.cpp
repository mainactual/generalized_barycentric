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
#include <iostream>

void extract( const cv::Mat & src, std::vector< cv::Mat > & dst );
void merge_images( const std::vector< cv::Mat > & in, cv::Mat & out, double factor );

template< class Precision > void get_circles(
	std::vector< std::vector< cv::Vec<Precision,2> > > & pnts,
	std::vector< std::vector< cv::Vec<Precision,3> > > & colors,
	const std::vector< int > radii,
	const std::vector< int > corners,
	int cv_colormap,
	const cv::Vec<Precision,2> & center )
{
	typedef cv::Vec<Precision,2> TVec;
	typedef cv::Vec<Precision,3> TCol;
	if ( radii.size()!=corners.size() )
		CV_Error(cv::Error::StsError,"empty");
	pnts.resize( radii.size() );
	colors.resize( radii.size() );
	// radii should be increasing or decreasing, not checking!
	size_t total=0;
	for ( int k=0;k<radii.size();++k )
	{
		double r = radii[k];
		std::vector< TVec > v;
		for ( int i=0, imax=corners[k]; i<imax;++i )
		{
			double angle = (double)i * 2.0 * CV_PI/(double)imax;
			TVec pos = center + TVec(r * cos( angle ), r * sin(angle) );
			v.push_back( pos );
		}
		if ( (k % 2) == 1 )
			std::reverse( v.begin(), v.end() );
		pnts[k] = v;
		total+=v.size();
	}
	// make colors from lut
	cv::Mat lut( 1, total, CV_8UC1 );
	for ( int x=0;x<lut.cols; ++x )
		lut.at<unsigned char>(x)=cv::saturate_cast<unsigned char>( std::round(255.0 * (double)x/(double)(lut.cols-1) ) );
	cv::Mat cols;
	cv::applyColorMap( lut, cols, cv_colormap );
	for ( size_t k=0, offset=0;k<radii.size(); ++k )
	{
		std::vector< TCol > c( pnts[k].size() );		
		for ( size_t i=0;i<c.size(); ++i, ++offset )
		{
			cv::Vec3b b = cols.at<cv::Vec3b>(offset);
			c[i] = TCol( b[0],b[1],b[2] );
		}
		colors[k] = c;
	}
}

template< class Precision > double colorfill_assert( cv::Size size, int seed )
{
	//std::cout << "Seed = " << seed << std::endl;
	typedef cv::Vec<Precision,2> TVec;
	typedef cv::Vec<Precision,3> TCol;

	int cvtype = tcv::get_cv_type<Precision>();
	int fulltype = CV_MAKE_TYPE( cvtype, 3 );

	
	cv::Mat m1,m2;
	std::srand( seed );
	typedef std::vector< TVec > V;
	typedef std::vector< TCol > C;
	std::vector< V > v;
	std::vector< C > c;
	get_circles<Precision>(v,c,
		std::vector<int>{size.width/3,size.width/4},
		std::vector<int>{ 3 + (std::rand()%61), 3 + (std::rand()%61) }, 
		cv::COLORMAP_RAINBOW,
		TVec( size.width/2, size.height/2 ) );
	std::vector< size_t > N(v.size() );
	for ( size_t i=0;i<v.size();++i )
	{
		N[i] = v[i].size();
		v[i] = padd_vector( v[i] );
		c[i] = padd_vector( c[i] );
	}

	m1 = cv::Mat( size, fulltype, cv::Scalar(0) );
	for ( int y=0;y<m1.rows;++y)
	{
		for ( int x=0;x<m1.cols;++x)
		{
			m1.at< TCol >(y,x) = calculate_weight_evaluate( c, v, TVec(x,y), N );
			//cv::Vec3b val;
			//for ( unsigned int u=0;u<3;++u )
			//	val[u]=cv::saturate_cast<unsigned char>(fval[u]);
		}
	}

	cv::Mat buffer;
	m2 = cv::Mat( size, fulltype, cv::Scalar(0) );
	for ( int y=0;y<m2.rows;++y)
	{
		for ( int x=0;x<m2.cols;++x)
		{
			m2.at< TCol >(y,x) = calculate_weight_evaluate( c, v, TVec(x,y), buffer, N );
		}
	}
	
	cv::Mat diff;
	cv::absdiff( m1, m2, diff );
	std::vector< cv::Mat > channels;
	extract( diff, channels );
	cv::Mat mask( diff.size(), CV_8UC1, cv::Scalar(0) );
	double maximum=0.0;
	for ( size_t i=0;i<channels.size(); ++i )
	{
		double m;
		cv::minMaxLoc( channels[i], 0, &m );
		cv::add( mask, channels[i]>10, mask );
		maximum = std::max( maximum, m );
	}
	return maximum;
	
}
double colorfill_assert128( cv::Size size, int seed )
{
	std::cout << "Seed = " << seed << std::endl;
	typedef cv::Vec2f TVec;
	typedef cv::Vec3f TCol;

	int fulltype = CV_32FC3;

	cv::Mat m1,m2;
	std::srand( seed );
	typedef std::vector< TVec > V;
	typedef std::vector< TCol > C;
	std::vector< V > v;
	std::vector< C > c;
	get_circles<float>(v,c,
		std::vector<int>{size.width/3,size.width/4 },
		std::vector<int>{ 3 + (std::rand()%61),  3 + (std::rand()%61) }, 
		cv::COLORMAP_RAINBOW,
		TVec( size.width/2, size.height/2 ) );
	std::vector< size_t > N(v.size() );
	for ( size_t i=0;i<v.size();++i )
	{
		N[i] = v[i].size();
		v[i] = padd_vector( v[i] );
		c[i] = padd_vector( c[i] );
	}
	
	double time1=0.0, time2=0.0;
	{
		cv::Mat buffer;
		m1 = cv::Mat( size, fulltype, cv::Scalar(0) );
		int64 t = cv::getTickCount();
		for ( int y=0;y<m1.rows;++y)
		{
			for ( int x=0;x<m1.cols;++x)
			{
				//m1.at< TCol >(y,x) = calculate_weight_evaluate<TCol,float >( c, v, TVec(x,y), N );
				m1.at< TCol >(y,x) = calculate_weight_evaluate<TCol,float>( c, v, TVec(x,y), buffer, N );
			}
		}
		t = cv::getTickCount()-t;
		time1 = (double)t/(double)cv::getTickFrequency();
		//std::cout << "time " << time1 << std::endl;
	}
	{
		m2 = cv::Mat( size, fulltype, cv::Scalar(0) );
		int64 t = cv::getTickCount();
		for ( int y=0;y<m2.rows;++y)
		{
			for ( int x=0;x<m2.cols;++x)
			{
				m2.at< TCol >(y,x) = calculate_weight_evaluate128( c, v, TVec(x,y), N );
			}
		}
		t = cv::getTickCount()-t;
		time2 = (double)t/(double)cv::getTickFrequency();
		//std::cout << "time SSE " << time2 << std::endl;
	}
	if ( !m2.empty() )
	{
		std::cout << "Speed: x" << time1/time2 << std::endl;
		cv::Mat diff;
		cv::absdiff( m1, m2, diff );
		std::vector< cv::Mat > channels;
		extract( diff, channels );
		cv::Mat mask( diff.size(), CV_8UC1, cv::Scalar(0) );
		double maximum=0.0;
		for ( size_t i=0;i<channels.size(); ++i )
		{
			double m;
			cv::minMaxLoc( channels[i], 0, &m );
			cv::add( mask, channels[i]>10, mask );
			maximum = std::max( maximum, m );
		}
		std::cout << "Max dif " << maximum << std::endl;
		if ( 1 )
		{
			cv::Mat mrgf, mrg;
			//m2.setTo( cv::Scalar(0,0,0), mask );
			merge_images( std::vector< cv::Mat >{m1,m2}, mrgf, 1.0 );
			mrgf.convertTo( mrg, CV_8U );
			cv::imshow( "win", mrg );
			cv::waitKey( 1000 );
		}
		return maximum;
	}
	return 0.0;	
}
void colorfill()
{
	double maximum = 0.0;
	cv::Size size(900,900);
	for ( int i=0;i<10;++i )
	{
		int seed = cv::getTickCount() % 100; 
		double d = colorfill_assert<double>( size, seed );
		maximum = std::max( d, maximum );
		colorfill_assert128( size, seed );
	}
	std::cout << "maximum diff " << maximum << std::endl;
}
void polygonfill()
{
	cv::Size size(400,400);
	typedef cv::Vec2f TVec;
	typedef cv::Vec3f TCol;
	int fulltype = CV_32FC3;
	typedef std::vector< TVec > V;
	typedef std::vector< TCol > C;
	std::vector< V > v;
	std::vector< C > c;
	cv::Mat m1,m2;
	get_circles<float>(v,c,
		std::vector<int>{size.width/3 },
		std::vector<int>{ 3 }, 
		cv::COLORMAP_RAINBOW,
		TVec( size.width/2, size.height/2 ) );
	std::vector< size_t > N(v.size() );
	for ( size_t i=0;i<v.size();++i )
	{
		N[i] = v[i].size();
		v[i] = padd_vector( v[i] );
		c[i] = padd_vector( c[i] );
	}
	
	m1 = cv::Mat( size, fulltype );
	for ( int y=0;y<m1.rows;++y)
	{
		for ( int x=0;x<m1.cols;++x)
		{
			m1.at< TCol >(y,x) = calculate_weight_evaluate_triangle( c[0], v[0], TVec(x,y), TCol(0,0,0) );
		}
	}
	cv::Mat buffer;
	m2 = cv::Mat( size, fulltype );
	for ( int y=0;y<m2.rows;++y)
	{
		for ( int x=0;x<m2.cols;++x)
		{
			m2.at< TCol >(y,x) = calculate_weight_evaluate128( c, v, TVec(x,y), N );
		}
	}

	cv::Mat m3;
	cv::subtract( m2,m1,m3 );
	cv::Mat mrgf, mrg;
	merge_images( std::vector< cv::Mat >{m1,m2,m3}, mrgf, 1.0 );
	mrgf.convertTo( mrg, CV_8U );
	cv::imshow( "win", mrg );
	//cv::imwrite( "D:/tmp/polygonfill.jpg", mrg );
	cv::waitKey( 0 );
}
