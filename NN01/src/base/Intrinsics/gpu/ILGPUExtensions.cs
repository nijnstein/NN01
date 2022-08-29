using ILGPU.Runtime;
using ILGPU;
using ILGPU.Algorithms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime.OpenCL;
using static NN01.ILGPUExtensions;

namespace NN01
{
    static public class ILGPUExtensions
    {
        static public bool HaveGPUAcceleration
        {
            get 
            {
                using (Context ctx = Context.CreateDefault())
                {
                    return ctx.Devices.Any(x => x.AcceleratorType != AcceleratorType.CPU);
                }
            }
        }


        static public GPUContext? CreateGPUContext()
        {
            Context context = Context.Create().Default().EnableAlgorithms().ToContext();

            Device? device = context.GetPreferredDevice(preferCPU: false);

            if (device == null)
            {
                return null; 
            }
            else
            {
                return new GPUContext()
                {
                    Context = context,
                    Device = device
                };
            }
        }

        
      
      //          var kernel = acc.LoadAutoGroupedStreamKernel<
      //              Index1D, ArrayView<float>, ArrayView<double>, ArrayView<double>>(MathKernel);
      //
      //
      //          var buffer = acc.Allocate1D<float>(128);
      //          var buffer2 = acc.Allocate1D<double>(128);
      //          var buffer3 = acc.Allocate1D<double>(128);
      //
      //          // Launch buffer.Length many threads
      //          kernel((int)buffer.Length, buffer.View, buffer2.View, buffer3.View);
      //
      //          // Wait for the kernel to finish...
      //          acc.Synchronize();
      //
      //          // Resolve and verify data
      //          var data = buffer.GetAsArray1D();
      //          var data2 = buffer2.GetAsArray1D();
      //          var data3 = buffer3.GetAsArray1D();
      //
      //          Console.WriteLine($"Math results: {data[10]} (float) {data2[10]} (double [GPUMath]) {data3[10]}");
      //
      //          buffer.Dispose();
      //          buffer2.Dispose();
      //          buffer3.Dispose();
      //          Console.WriteLine("CUDA test complete");
            

    } 
}
