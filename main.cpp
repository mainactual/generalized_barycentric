#include <iostream>
#include "opencv2/core.hpp"

void imagewarp();
void colorfill();
void polygonfill();

int main( int argc, char * argv [] )
{
	try
	{
		colorfill();

	}catch( std::exception & err )
	{
		std::cout << err.what() << std::endl;
	}
	return 0;
}
