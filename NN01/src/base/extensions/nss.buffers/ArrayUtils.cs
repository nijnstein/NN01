using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NSS
{
    public static class ArrayUtils
    {

        public static T[,] ConvertTo2D<T>(this T[][] a)
        {
            Debug.Assert(a != null && a.Length > 0 && a[0].Length > 0); 
            Debug.Assert(a.All(x => x.Length == a[0].Length)); // dont skip the first .. collection of 1 

            int rows = a.Length;
            int columns = a[0].Length;

            T[,] b = new T[rows, columns];

            unchecked
            {
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < columns; c++)
                        b[r, c] = a[r][c];
            }

            return b; 
        }

    }
}
