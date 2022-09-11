using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NSS
{

    /// <summary>
    /// the random number generator used on the gpu 
    /// </summary>
    public enum RandomProviderBackend
    {
        XorShift128,
        XorShift128Plus,
        XorShift32,
        XorShift64star,
    }
}
