namespace NSS
{
    public struct Size2D
    {
        public int X;
        public int Y; 
        public Size2D (int x, int y)
        {
            X = x;
            Y = y; 
        }

        public static Size2D Zero = new Size2D(0, 0); 
    }
}
