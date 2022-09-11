using ILGPU;
using ILGPU.Algorithms.Random;
using ILGPU.IR.Values;
using ILGPU.Runtime;
using NN01;
using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;

namespace NN01 
{
    public static class DistributionKernel
    {
        public static Action<Index1D, ArrayView<float>, float, float> GetGPU(LayerInitializationType distribution)
        {
            switch (distribution)
            {
                case LayerInitializationType.Default:
                case LayerInitializationType.Random: return RandomGPUKernel;

                case LayerInitializationType.Zeros: return ZeroFillGPUKernel;
                case LayerInitializationType.Ones: return OneFillGPUKernel;
                case LayerInitializationType.Uniform: return UniformGPUKernel;
                case LayerInitializationType.Normal: return NormalGPUKernel;

                default:
                    throw new InvalidOperationException("heNormal distribution type should have been converted into a normal kernel");
            }
        }
        public static Action<Memory<float>, float, float> GetCPU(LayerInitializationType distribution)
        {
            switch (distribution)
            {
                case LayerInitializationType.Default:
                case LayerInitializationType.Random: return RandomCPUKernel;

                case LayerInitializationType.Zeros: return ZeroFillCPUKernel;
                case LayerInitializationType.Ones: return OneFillCPUKernel;
                case LayerInitializationType.Uniform: return UniformCPUKernel;
                case LayerInitializationType.Normal: return NormalCPUKernel;

                default:
                    throw new InvalidOperationException("heNormal distribution type should have been converted into a normal kernel");
            }
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void ZeroFillGPUKernel(Index1D index, ArrayView<float> data, float p1, float p2) => data[index] = 0f;
    
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void OneFillGPUKernel(Index1D index, ArrayView<float> data, float p1, float p2) => data[index] = 1f;
        
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void RandomGPUKernel(Index1D index, ArrayView<float> data, float p1, float p2)
        {
            data[index] = (p2 - p1) * data[index] - (p1 + p2) / 2;
        }
        
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void NormalGPUKernel(Index1D index, ArrayView<float> data, float mean, float sd)
        {
            float x = data[index];   
            float p = 1f / MathF.Sqrt(2f * MathF.PI * (sd * sd));
            data[index] = (p * MathF.Exp(-0.5f * ((x - mean) * (x - mean)) / (sd * sd)));
        }
        
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void GaussianGPUKernel(Index1D index, ArrayView<float> data, float mean, float sd)
        {
            float x1 = 1f - data[index];
            float x2 = 1f - data[data.Length - index - 1];
            float y1 = MathF.Sqrt(-2f * MathF.Log(x1)) * MathF.Cos(2f * (float)Math.PI * x2);
            data[index] = y1 * sd + mean;
        }
        
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void UniformGPUKernel(Index1D index, ArrayView<float> data, float p1, float p2) => data[index] = (p2 - p1) / data.Length + p1;

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void ZeroFillCPUKernel(Memory<float> memory, float p1, float p2)
        {
            unchecked
            {
                Span<float> data = memory.Span; 
                int i = 0; 
                while(i < (data.Length & ~3))
                {  
                    data[i + 0] = 0;
                    data[i + 1] = 0;
                    data[i + 2] = 0;
                    data[i + 3] = 0;
                    data[i + 4] = 0;
                    data[i + 5] = 0;
                    data[i + 6] = 0;
                    data[i + 7] = 0;
                    i += 8;
                }
                while (i < data.Length)
                {
                    data[i++] = 0;
                }
            }
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void OneFillCPUKernel(Memory<float> memory, float p1, float p2)
        {
            unchecked
            {
                Span<float> data = memory.Span; 
                int i = 0;
                while (i < (data.Length & ~3))
                {
                    data[i + 0] = 0;
                    data[i + 1] = 0;
                    data[i + 2] = 0;
                    data[i + 3] = 0;
                    data[i + 4] = 0;
                    data[i + 5] = 0;
                    data[i + 6] = 0;
                    data[i + 7] = 0;
                    i += 8;
                }
                while (i < data.Length)
                {
                    data[i++] = 0;
                }
            }
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void RandomCPUKernel(Memory<float> memory, float p1, float p2)
        {
            Span<float> data = memory.Span;
            if (p1 == 0f && p2 == 1f)
            {
                unchecked
                {
                    for (int i = 0; i < data.Length; i++)
                    {
                        data[i] = Random.Shared.NextSingle() - 0.5f;
                    }
                }
            }
            else
            {
                float a = (p2 - p1);
                float b = (p1 + p2) / 2;
                for (int i = 0; i < data.Length; i++)
                {
                    data[i] = a * Random.Shared.NextSingle() - b;
                }
            }
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void NormalCPUKernel(Memory<float> memory, float sd, float mean)
        {
            Span<float> data = memory.Span;
            unchecked
            {
                for(int i = 0; i < data.Length; i++)
                {
                    float x = Random.Shared.NextSingle();
                    float p = 1f / MathF.Sqrt(2f * MathF.PI * (sd * sd));
                    data[i] = p * MathF.Exp(-0.5f * ((x - mean) * (x - mean)) / (sd * sd));
                }
            }
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void GaussianCPUKernel(Memory<float> memory, float sd, float mean)
        {
            Span<float> data = memory.Span;
            unchecked
            {
                for (int i = 0; i < data.Length; i++)
                {
                    float x1 = 1 - Random.Shared.NextSingle();
                    float x2 = 1 - Random.Shared.NextSingle();
                    float y1 = MathF.Sqrt(-2f * MathF.Log(x1)) * MathF.Cos(2f * (float)Math.PI * x2);
                    data[i] = y1 * sd + mean;
                }
            }
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void UniformCPUKernel(Memory<float> memory, float low, float high)
        {
            Span<float> data = memory.Span;
            unchecked
            {
                for (int i = 0; i < data.Length; i++)
                {
                    data[i] = (high - low) / data.Length + low;
                }
            }
        }
    }
}
