using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.OpenCL;

namespace NSS.GPU
{
    public class GPUContext : IDisposable
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

        static public GPUContext? Create()
        {
            Context context = Context.Create()
                .Default()
                .EnableAlgorithms()
                .Caching(CachingMode.Default)
                .Inlining(InliningMode.Aggressive)
                .DebugSymbols(DebugSymbolsMode.Disabled)
                .Optimize(OptimizationLevel.O2)
                .Math(MathMode.Fast32BitOnly).ToContext();

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

        public Context? Context { get; set; }
        public Device? Device { get; set; }
        public Accelerator CreateCPUAccelerator() => Context.CreateCPUAccelerator(0);
        public Accelerator CreateGPUAccelerator() => Device!.AcceleratorType == AcceleratorType.Cuda ? CreateCudaAccelerator() : CreateOpenCLAccelerator(); 
        public Accelerator CreateOpenCLAccelerator() => Context.CreateCLAccelerator(0);
        public Accelerator CreateCudaAccelerator() => Context.CreateCudaAccelerator(0);
        public void Dispose()
        {
            if (Context != null)
            {
                Context.Dispose();
            }
            Device = null;
            Context = null;                 
        }
    }   
}
