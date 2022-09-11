using NSS;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;

namespace NN01 
{
    public static class ActivationFunctions
    {
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Identity(float x) => x;

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float IdentityDerivative(float x) => x;

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sigmoid(this float f)
        {
            // faster alternative:
            // see https://stackoverflow.com/questions/412019/math-optimization-in-c-sharp 

            // fast sigmoid calculation (but the same curve) 
            // return .5f + .5f * TanhFast(f * .5f); 
            f *= 0.5f;
            return 0.5f + 0.5f * (f / (1f + MathF.Abs(f)));

            // normal sigmoid calculation: 
            // f = MathF.Exp(f);
            //  return f / (1.0f + f);
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SigmoidDerivative(this float f)
        {
            return f * (1f - f);
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Swish(this float f)
        {
            return f * f.Sigmoid();
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SwishDerivative(this float f)
        {
            // see
            // https://sefiks.com/2018/08/21/swish-as-neural-networks-activation-function/
            float e = 1f + MathF.Exp(-f);
            return (f * e + (e - f)) / (e * e);
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float TanhFast(this float x)
        {
            return x / (1f + MathF.Abs(x));
            // return (float)MathF.Tanh(x);
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float TanhFastDerivative(float x)
        {
            return 1f - x * x;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Tanh(float x)
        {
            float x2e = MathF.Exp(2f * x);
            return (x2e - 1f) / (x2e + 1f);
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float TanhDerivative(float x)
        {
            return 4f / (MathF.Exp(-x) + MathF.Exp(x)).Square();
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LeCunTanh(float x)
        {
            const float divX = 2f / 3f;
            const float scale = 1.7159f;

            float e2x = (float)MathF.Exp(2f * divX * x);
            return scale * (e2x - 1f) / (e2x + 1f);
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LeCunTanhPrime(float x)
        {
            float exp = 2f * x / 3f;
            return 4.57573f / (MathF.Exp(exp) + MathF.Exp(-exp)).Square();
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ReLU(this float x)
        {
            return x < 0 ? 0 : x;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ReLUDerivative(float x)
        {
            return x < 0 ? 0 : 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LeakyReLU(this float x)
        {
            return x < 0 ? 0.01f * x : x;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LeakyReLUDerivative(float x)
        {
            return x < 0 ? 0.01f : 1;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ELU(float x) => x < 0f ? MathF.Exp(x) - 1f : x;

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ELUDerivative(float x) => x < 0f ? MathF.Exp(x) : 1f;

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float AbsReLU(float x) => x < 0 ? -x : x;

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float AbsReLUDerivate(float x) => x < 0 ? -1 : 1;


        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Softplus(float x)
        {
            // ln(1 + e^x)
            // the derivative is sigmoid(x)
            return MathF.Log(1 + MathF.Exp(x));
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SoftplusDerivative(float x) => x.Sigmoid();


        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ParametricReLU(float x, float a)
        {
            return x < 0 ? a * x : x;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ParametricReLUDerivative(float x, float a)
        {
            return x < 0 ? a : 1;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Logistic(float x)
        {
            return 1f / (1f + MathF.Pow(MathF.E, -x));
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LogisticDerivative(float x)
        {
            return Logistic(x) * (1f - Logistic(x));
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float BipolarSigmoid(float x)
        {
            return (1f - MathF.Exp(-x)) / (1F + MathF.Exp(-x));
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float BipolarSigmoidDerivative(float x)
        {
            return 0.5f * (1f + BipolarSigmoid(x)) * (1f - BipolarSigmoid(x));
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Gaussian(float x)
        {
            return MathF.Pow(MathF.E, MathF.Pow(-x, 2f));
        }
        
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float GaussianDerivative(float x)
        {
            return -2f * x * MathF.Pow(MathF.E, MathF.Pow(-x, 2f));
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float BentIdentity(float x)
        {
            return (((MathF.Sqrt(MathF.Pow(x, 2f) + 1f)) - 1f) / 2f) + x;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DBentIdentity(float x)
        {
            return (x / (2f * MathF.Sqrt(MathF.Pow(x, 2f) + 1f))) + 1f;
        }
    }
}
