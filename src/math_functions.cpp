/*
*
* This file is part of the open-source SeetaFace engine, which includes three modules:
* SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
*
* This file is part of the SeetaFace Identification module, containing codes implementing the
* face identification method described in the following paper:
*
*
*   VIPLFaceNet: An Open Source Deep Face Recognition SDK,
*   Xin Liu, Meina Kan, Wanglong Wu, Shiguang Shan, Xilin Chen.
*   In Frontiers of Computer Science.
*
*
* Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
* Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
*
* The codes are mainly developed by Zining Xu(a M.S. supervised by Prof. Shiguang Shan)
*
* As an open-source face recognition engine: you can redistribute SeetaFace source codes
* and/or modify it under the terms of the BSD 2-Clause License.
*
* You should have received a copy of the BSD 2-Clause License along with the software.
* If not, see < https://opensource.org/licenses/BSD-2-Clause>.
*
* Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
*
* Note: the above information must be kept whenever or wherever the codes are used.
*
*/

#include "math_functions.h"

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif


// simd_dot采用了SIMD技术
// 注意：simd_dot函数在其他多个文件中也有被引用（条件分支），所以不要更改这个函数的名字
float simd_dot(const float* x, const float* y, const long& len) {
	float inner_prod = 0.0f;
	__m128 X, Y; // 128-bit values
	__m128 acc = _mm_setzero_ps(); // set to (0, 0, 0, 0)
	float temp[4];

	long i;
	for (i = 0; i + 4 < len; i += 4) {
		X = _mm_loadu_ps(x + i); // load chunk of 4 floats
		Y = _mm_loadu_ps(y + i);
		acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));
	}
	_mm_storeu_ps(&temp[0], acc); // store acc into an array
	inner_prod = temp[0] + temp[1] + temp[2] + temp[3];

	// add the remaining values, 上面的SIMD一次处理4个数的运算，下面处理余数
	for (; i < len; ++i) {
		inner_prod += x[i] * y[i];
	}
	return inner_prod;
}

// matrix_procuct的输入分别为：输入数据首地址A、权重数据首地址B、输出数据首地址C、ofmap平面元素数量n、
// output volume的channel数m、一个filter的元素数量k。。。
void matrix_procuct(const float* A, const float* B, float* C, const int n,
	const int m, const int k, bool ta, bool tb) {
	const float* x = B;
	for (int i = 0, idx = 0; i < m; ++i) {	// 循环完毕将产生完整的output volume
		const float* y = A;
		for (int j = 0; j < n; ++j, ++idx) {	// 乘加运算：执行一次该循环将产生一个ofmap元素，循环完毕将产生output volume中的一张ofmap
			C[idx] = simd_dot(x, y, k);
			y += k;
		}
		x += k;
	}	
}

