/*
 * run_all_test.cpp

 *
 *  Created on: 11/06/2015
 *      Author: bsp
 */

#include <stdio.h>
#include "gtest/gtest.h"




TEST(Dummy, Dummy)
{
	EXPECT_TRUE(true);
}

int main(int argc, char** argv)
{
 testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();

}


