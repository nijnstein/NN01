using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public class Pattern
    {
        public float[] Data { get; set; }
        public int Class { get; set; }
        public int Length => Data.Length;
    }

    public class Pattern2D : Pattern
    {
        public int Width { get; set; }
        public int Height { get; set; }
    }
}
