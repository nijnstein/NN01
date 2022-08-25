using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public static class ActivationFunctions
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sigmoid(this float f)
        {
            f = MathF.Exp(f);
            return f / (1.0f + f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SigmoidDerivative(this float f)
        {
            return f * (1f - f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Swish(this float f)
        {
            return f * f.Sigmoid();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SwishDerivative(this float f)
        {
            // see
            // https://sefiks.com/2018/08/21/swish-as-neural-networks-activation-function/
            float e = 1f + MathF.Exp(-f);
            return (f * e + (e - f)) / (e * e);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Tanh(this float x)
        {
            return (float)MathF.Tanh(x);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float TanhDerivative(float x)
        {
            return 1 - (x * x);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ReLU(this float x)
        {
            return x < 0 ? 0 : x;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LeakyReLUDerivative(float x)
        {
            return x < 0 ? 0.01f : 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LeakyReLU(this float x)
        {
            return x < 0 ? 0.01f * x : x;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ReLUDerivative(float x)
        {
            return x < 0 ? 0 : 1;
        }


    }
}
