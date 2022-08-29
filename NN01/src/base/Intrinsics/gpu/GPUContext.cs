using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public class GPUContext : IDisposable
    {
        static public bool HaveGPUAcceleration => ILGPUExtensions.HaveGPUAcceleration; 

        public Context Context { get; set; }
        public Device Device { get; set; }
        public Accelerator CreateAccelerator() => Device.CreateAccelerator(Context);
        public void Dispose()
        {
            if (Context != null)
            {
                Context.Dispose();
            }
            Device = null;
            Context = null;                 
        }

        public static GPUContext Create() => ILGPUExtensions.CreateGPUContext(); 
    }   
}
