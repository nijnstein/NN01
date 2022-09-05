using System.Drawing;

namespace NSS.Imaging
{
    /// <summary>
    /// interface to element list 
    /// </summary>
    public interface IElementList : IEnumerable<IElement>
    {
        int ElementCount { get; }
        double PageWidth { get; }
        double PageHeight { get; }
        int Confidence { get; }

        Rectangle ContentBorder { get; }

        void Add(IElement element);
        bool Remove(IElement element);
    }

    /// <summary>
    /// abstract base for all element lists 
    /// </summary>
    public abstract class BaseElementList : IElementList
    {
        protected List<IElement> m_Elements = new List<IElement>();
        protected Rectangle m_ContentBorder = Rectangle.Empty;
        protected bool m_Changed = false;
        protected double m_PageWidth = 0;
        protected double m_PageHeight = 0;

        public object Tag { get; set; }
    
        static public int TopLeft2RightBottomSorter(IElement x, IElement y)
        {
            if (x == null && y == null) return 0;
            if (x == null) return -1;
            if (y == null) return 1;

            if (x.Region.Top < y.Region.Top) return -1;
            else
             if (x.Region.Top > y.Region.Top) return 1;
            else
              if (x.Region.Left < y.Region.Left) return -1;
            else
               if (x.Region.Left > y.Region.Left) return 1;
            else
                return 0;
        }


        #region properties
        public double PageWidth
        {
            get
            {
                return m_PageWidth;
            }
            set
            {
                Changed = true;
                m_PageWidth = value;
            }
        }

        public double PageHeight
        {
            get
            {
                return m_PageHeight;
            }
            set
            {
                Changed = true;
                m_PageHeight = value;
            }
        }

        public int ElementCount
        {
            get
            {
                return m_Elements == null ? 0 : m_Elements.Count;
            }
        }

        virtual public int WordCount
        {
            get
            {
                int c = 0;
                foreach (IElement e in this)
                    if (e.ElementType == ElementType.Word)
                        c++;
                return c;
            }
        }

        public bool Changed
        {
            get
            {
                return m_Changed;
            }
            set
            {
                m_Changed = value;
                if (m_Changed == true)
                {
                    Invalidate();
                }
            }
        }

        public virtual void Invalidate()
        {
            // m_Changed = false; 
            m_ContentBorder = Rectangle.Empty;
        }

        public Rectangle ContentBorder
        {
            get
            {
                if (m_ContentBorder == Rectangle.Empty && ElementCount > 0)
                {
                    m_ContentBorder = CalcOuterBorder(m_Elements);
                }
                return m_ContentBorder;
            }
        }

        public static Rectangle CalcOuterBorder(IEnumerable<IElement> elements)
        {
            if (elements == null || elements.Count() == 0) return Rectangle.Empty;
            else
            {
                int x = int.MaxValue;
                int y = int.MaxValue;
                int x2 = int.MinValue;
                int y2 = int.MinValue;

                foreach (IElement e in elements)
                {
                    if (e != null)
                    {
                        x = Math.Min(x, e.Region.Left);
                        x2 = Math.Max(x2, e.Region.Right);
                        y = Math.Min(y, e.Region.Top);
                        y2 = Math.Max(y2, e.Region.Bottom);
                    }
                }

                if (x > x2)
                {
                    int t = x;
                    x = x2;
                    x2 = t;
                }

                if (y > y2)
                {
                    int t = y;
                    y = y2;
                    y2 = t;
                }

                return Rectangle.FromLTRB(x, y, x2, y2);
            }
        }

        /// <summary>
        /// confidence factor 0-100 
        /// </summary>
        public int Confidence
        {
            get
            {
                if (ElementCount == 0)
                {
                    return 100;
                }
                else
                {
                    int t = 0;
                    foreach (IElement e in this)
                        t += e.Confidence;
                    return t / ElementCount;
                }
            }
        }
        #endregion

        #region IEnumerable<IElement> Members

        public IEnumerator<IElement> GetEnumerator()
        {
            return m_Elements.GetEnumerator();
        }

        #endregion

        #region IEnumerable Members

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return m_Elements.GetEnumerator();
        }

        #endregion

        #region list actions

        virtual public void Add(IElement element)
        {
            if (m_Elements == null) m_Elements = new List<IElement>();
            m_Elements.Add(element);
            Changed = true;
        }
        virtual public void Add(IEnumerable<IElement> elements)
        {
            if (m_Elements == null) m_Elements = new List<IElement>();
            m_Elements.AddRange(elements);
            Changed = true;
        }
        virtual public bool Remove(IElement element)
        {
            if (ElementCount == 0) return false;
            else
            {
                if (m_Elements.Remove(element))
                {
                    Changed = true;
                    return true;
                }
                else
                {
                    return false;
                }
            }
        }

        virtual public void Clear()
        {
            if (ElementCount == 0) return;
            else
            {
                m_Elements.Clear();
                Changed = true;
                return;
            }
        }

        #endregion

        #region directional element scan

        virtual public IElement Scan(int from_x, int from_y, int to_x, int to_y, int w_search, int h_search, int max_h, IEnumerable<IElement> exclude)
        {
            if (ElementCount == 0) return null;

            int dx = to_x - from_x;
            int dy = to_y - from_y;

            // TODO somebody who can actually calculate can do this a lot better... 
            double l = Math.Sqrt(dx * dx + dy * dy);
            double l_step = l / 50.0;

            for (double pos = 0.0; pos <= 1.0; pos += 0.01)
            {
                double x1 = from_x + dx * pos - w_search / 2;
                double y1 = from_y + dy * pos - h_search / 2;

                Rectangle rc = new Rectangle((int)x1, (int)y1, w_search, h_search);

                foreach (IElement element in m_Elements)
                {
                    if (max_h == 0 || element.Region.Height < max_h * 4)
                    {
                        if (rc.IntersectsWith(element.Region) || rc.Contains(element.Region))
                        {
                            if (exclude != null)
                            {
                                bool bexclude = false;
                                foreach (IElement e in exclude)
                                    if (e == element)
                                    {
                                        bexclude = true;
                                        break;
                                    }
                                if (bexclude) continue;
                            }
                            return element;
                        }
                    }
                }
            }
            return null;
        }

        public IElement ElementToRight(IElement from)
        {
            return ElementToRight(from, ContentBorder);
        }

        public IElement ElementToRight(IElement from, Rectangle rc_content)
        {
            int max_h = (int)(from.Region.Height * 1.4);
            int x = from.Region.Right + 1;
            int y = from.Region.Top + from.Region.Height / 2;

            IElement element = Scan
            (
             x, y,
             rc_content.Right, y,
             10,
             from.Region.Height / 2,
             max_h,
             new IElement[1] { from }
            );

            return element;
        }

        public IElement ElementToLeft(IElement from)
        {
            return ElementToLeft(from, ContentBorder);
        }

        public IElement ElementToLeft(IElement from, Rectangle rc_content)
        {
            int max_h = (int)(from.Region.Height * 1.4);
            int x = from.Region.Left - 1;
            int y = from.Region.Top + from.Region.Height / 2;

            IElement element = Scan
            (
             x, y,
             rc_content.Left, y,
             10,
             from.Region.Height / 2,
             max_h,
             new IElement[1] { from }
            );

            return element;
        }

        public IElement ElementToBottom(IElement from)
        {
            return ElementToBottom(from, ContentBorder);
        }

        public IElement ElementToBottom(IElement from, Rectangle rc_content)
        {
            int max_h = (int)(from.Region.Height * 1.4);
            int x = from.Region.Left + from.Region.Width / 2;
            int y = from.Region.Bottom + 1;

            IElement element = Scan
            (
             x, y,
             x, rc_content.Bottom,
             from.Region.Width,
             from.Region.Height / 2,
             max_h,
             new IElement[1] { from }
            );

            return element;
        }

        #endregion

        #region WordsInRect
        public List<IElement> WordsInRect(int x1, int y1, int x2, int y2)
        {
            if (ElementCount == 0) return new List<IElement>();
            return WordsInRect(new Rectangle(x1, y1, x2 - x1, y2 - y1));
        }
        public List<IElement> WordsInRect(Rectangle rect)
        {
            List<IElement> l = new List<IElement>();
            if (ElementCount == 0) return l;

            foreach (IElement e in m_Elements)
            {
                if (rect.IntersectsWith(e.Region))
                {
                    l.Add(e);
                }
            }

            return l;
        }
        #endregion

        #region Element Growing

        public List<IElement> GrowH(IElement e)
        {
            return GrowH(e, null);
        }
        public List<IElement> GrowH(IElement element, List<IElement> ignore)
        {
            if (element == null) return null;

            double max_y = element.Region.Height * 1.2;
            double w_step = element.Region.Height * 3;

            return Grow(element, ignore, max_y, w_step, null);
        }

        public List<IElement> Grow(IElement element, List<IElement> ignore, double max_y, double w_step, object stop)
        {
            if (element == null) return null;

            if (max_y == 0) max_y = element.Region.Height;
            double wleft = w_step / 2;

            List<IElement> l = new List<IElement>();
            l.Add(element);

            bool growing = true;
            Rectangle rc = new Rectangle(element.Region.Left, element.Region.Top, element.Region.Width, element.Region.Height);

            while (growing)
            {
                growing = false;

                Rectangle rc_grow = new Rectangle(rc.Left - (int)wleft, rc.Top, rc.Width + (int)w_step + (int)wleft, rc.Height);
                List<IElement> l2 = WordsInRect(rc_grow);

                foreach (IElement other in l2)
                {
                    if (!l.Contains(other) && (ignore == null || !ignore.Contains(other)))
                    {
                        int x1 = Math.Min(rc.Left, other.Region.Left);
                        int x2 = Math.Max(rc.Right, other.Region.Right);
                        int y1, y2;

                        if (other.Region.Height > max_y)
                        {
                            if (other.Region.Height > max_y * 3) continue;
                            y1 = Math.Min(rc.Top, other.Region.Top);
                            y2 = Math.Max(rc.Bottom, other.Region.Bottom);
                        }
                        else
                        {
                            y1 = rc.Top;
                            y2 = rc.Bottom;
                        }

                        if (other.Region.Left < rc.Left)
                            l.Insert(0, other);
                        else
                            l.Add(other);

                        if (stop != null)
                        {
                            if (other.CompareTo(stop, 95) > 0) continue; // same as return l; .. 
                                                                         // if (string.Compare(other.Text, stop, true) == 0 || string.Compare(other.Line.ToString(), stop, true) == 0)
                                                                         // continue;
                        }
                        rc = new Rectangle(x1, y1, x2 - x1, y2 - y1);
                        growing = true;
                    }
                }
            }

            return l;
        }

        #endregion

    }
}
