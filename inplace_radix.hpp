#ifndef INPLACE_RADIX_HPP
#define INPLACE_RADIX_HPP

#include "viennacl/context.hpp"
#include "viennacl/vector.hpp"
#include <sstream>
#include <stdint.h>

namespace ns_inplace_radix_select
{
	struct generate_value_type
	{
		template <typename T>
		std::string type()
		{
			return T::unimplemented;
		}

		template <>
		std::string type<uint32_t>()
		{
			return "#define VALUE_TYPE uint\n";
		}


		template <>
		std::string type<uint64_t>()
		{
			return "#define VALUE_TYPE uint\n";
		}

		template <>
		std::string type < double >()
		{
			return "#define VALUE_TYPE double\n";
		}

		template <>
		std::string type<float>()
		{
			return "#define VALUE_TYPE float\n";
		}

	};




	struct generate_extract_char_kernel
	{
		template<typename basic_type>
		std::string type()
		{
			return "inline uint extract_char(VALUE_TYPE val, uchar mask, uint shift)\n"
				"{\n"
				//long test = as_long(val);
				//long res = test ^ (-(test >> 63) | 0x8000000000000000 );
				"		return (val >> shift) & mask;\n"
				"}\n";
		}

		template <>
		std::string type<double>()
		{
			return "inline uint extract_char(double val, uchar mask, uint shift)\n"
				"{\n"
				" long test = as_long(val);\n "
				" long res = test ^ (-(test >> 63) | 0x8000000000000000 );\n"
				"		return (res >> shift) & mask;\n"
				"}\n";
		}

		template <>
		std::string type<float>()
		{
			return "inline uint extract_char(float val, uchar mask, uint shift)\n"
				"{\n"
				" uint res = as_uint(val);\n "
				"		return (res >> shift) & mask;\n"
				"}\n";
		}


	};
}

#endif