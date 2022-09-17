using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public static class LayerTypeExtensions
    {
        public static bool HasParameters(this LayerType type)
        {
            switch (type)
            {
                case LayerType.Dropout:
                case LayerType.Softmax:
                case LayerType.None: return false;

                default: return true;
            }
        }
    }
}
