#include <gtest/gtest.h>
#include "base_kernel_includes_nonisotropic.h"
#include "anisotropic_kernel.h"
#include "anisotropic_kernel.hpp"
#include "kernel_wenland_c2_anisotropic.h"
#include "sphinxsys.h"

using namespace SPH;

TEST(test_anisotropic_kernel, test_Laplacian)
{
	int y_num = 10; // particle number in y direction
	Real PH = 1.0;
	Real ratio_ = 4.0; //  dp_x /dp_y
	Real PL = ratio_* PH;

	Vec2d scaling_vector = Vec2d(1.0, 1.0 / ratio_);
	Real resolution_ref = PH / Real(y_num);
	Real resolution_ref_large = ratio_ * resolution_ref;
	Real V_ = resolution_ref * resolution_ref_large;
	Vec2d center = Vec2d(resolution_ref_large * 5.0, resolution_ref * 5.0);

	int x_num = PL / resolution_ref_large;
 
 	AnisotropicKernel<Anisotropic::KernelWendlandC2>  
 	    wendland(1.15 * resolution_ref_large, scaling_vector,  Vec2d(0.0, 0.0));
 	 
 	 	
	Real sum = 0.0;
	Vec2d first_order_rate = Vec2d(0.0, 0.0);
	Real second_order_rate =  0.0;
	
	 for (int i = 0; i < (x_num + 1); i++)
	{
		for (int j = 0; j < (y_num + 1); j++)
		{
			Real x = i * resolution_ref_large;
			Real y = j * resolution_ref;
			Vec2d displacement =  center - Vec2d(x, y) ;
			Real distance_ = displacement.norm();

			Real  sarutration_y =  y + x ;
			Real  sarutration_center_y =  center[1] + center[0];

			Real  sarutration_x =  x * x +  y * y;
			Real  sarutration_center_x = center[0] * center[0] + center[1] * center[1];
 
			if (wendland.checkIfWithinCutOffRadius(displacement))
			{
				Vec2d  eij_dwij_V = wendland.e(distance_, displacement)* wendland.dW(distance_, displacement) * V_;
		 
				sum += wendland.W(distance_, displacement)* V_;

				first_order_rate -= (sarutration_center_y - sarutration_y)* eij_dwij_V;

				second_order_rate +=  2.0 * (sarutration_center_x - sarutration_x)  
									* displacement.dot(wendland.e(distance_, displacement))  * V_ 
					 				* wendland.dW(distance_, displacement)  /  (distance_ + TinyReal)/ (distance_ + TinyReal); 
			
			}		
		}

	}



	EXPECT_EQ(1.0, sum); 
	EXPECT_EQ(1.0, first_order_rate[0]);
	EXPECT_EQ(1.0, first_order_rate[1]);
	EXPECT_EQ(2.0, second_order_rate);

}
 
int main(int argc, char* argv[])
{	
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
