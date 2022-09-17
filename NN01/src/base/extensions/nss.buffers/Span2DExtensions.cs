using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NSS
{
    public static class Span2DExtensions
    {

        /// <summary>
        /// CrossCorrelate a 2D map using a 2D kernel 
        /// 
        ///   0 1 2         0 1      19 25
        ///   3 4 5    *    2 3  =   37 43
        ///   6 7 8
        ///   
        ///  convolve a kernel over each ij in input and sum elements into a new map 
        ///  
        ///  new map size = mapsize[i - 1, j - 1]
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="kernel"></param>
        /// <param name="output"></param>


        static public void CrossCorrelate(this Span2D<float> input, Span2D<float> kernel, Span2D<float> output)
        {
            // iterate the kernel
            for (int y = 0; y < input.Height - kernel.Height + 1; y++)
            {
                for (int x = 0; x < input.Width - kernel.Width + 1; x++)
                {
                    float sum = 0;

                    for (int ky = 0; ky < kernel.Height; ky++)
                    {
                        for (int kx = 0; kx < kernel.Width; kx++)
                        {
                            sum += input[y + ky, x + kx] * kernel[ky, kx];
                        }
                    }

                    output[y, x] = sum; 
                }
            }
        }


    }
}
