using System.Drawing;

namespace NSS.Imaging
{
    /// <summary>
    /// possible elements
    /// </summary>
    public enum ElementType
    {
        /// <summary>
        /// no classification has yet been attempted 
        /// </summary>
        Undetermined,
        /// <summary>
        /// unknown element, classification failed on it.. 
        /// </summary>
        Unknown,
        /// <summary>
        /// proven text data, either through source (pdf) or OCR methods 
        /// </summary>
        Word,
        /// <summary>
        /// piece that might be text as classified by object mapper 
        /// </summary>
        Text,
        /// <summary>
        /// piece classified as textbox
        /// </summary>
        Checkbox,
        /// <summary>
        /// vertical, horizontal or diagonal line
        /// </summary>
        Line,
        /// <summary>
        /// box-frame, used for segmenting word collections and therefore template item matches, ie customerinfo like
        /// </summary>
        Frame,
        /// <summary>
        /// solid region, high density image or filled rectangle 
        /// </summary>
        Solid,
        /// <summary>
        /// something like an image .. 
        /// </summary>
        Image
    }


    /// <summary>
    /// base element interface 
    /// </summary>
    public interface IElement
    {
        /// <summary>
        /// type of element 
        /// </summary>
        ElementType ElementType { get; }

        /// <summary>
        /// object this element was based on 
        /// </summary>
        MapObject MapObject { get; }

        /// <summary>
        /// region of element (base element when children are involved)
        /// </summary>
        Rectangle Region { get; set; }
        /// <summary>
        /// region of element including children 
        /// </summary>
        Rectangle ContentRegion { get; }

        /// <summary>
        /// children belonging to this element
        /// </summary>
        IEnumerable<IElement> Children { get; }

        /// <summary>
        /// confidence factor, 0-100, 100 == perfect
        /// </summary>
        int Confidence { get; }

        /// <summary>
        /// Image data for the section (optional)
        /// </summary>
        Bitmap Image { get; set; }


        /// <summary>
        /// child count
        /// </summary>
        int ChildCount { get; }

        /// <summary>
        /// unique id for element within session
        /// </summary>
        int ID { get; }
        object Tag { get; }

        /// <summary>
        /// add a child, invalidating content region 
        /// </summary>
        /// <param name="e"></param>
        void AddChild(IElement e);
        /// <summary>
        /// add a list of children 
        /// </summary>
        /// <param name="list"></param>
        void AddChild(IEnumerable<IElement> list);

        /// <summary>
        /// remove a child element 
        /// </summary>
        /// <param name="e2"></param>
        void RemoveChild(IElement e2);

        /// <summary>
        /// compare this element to another element using a likeness factor
        /// </summary>
        /// <param name="element"></param>
        /// <param name="likeness">0-100</param>
        /// <returns></returns>
        int CompareTo(IElement element, int likeness);

        /// <summary>
        /// compare this element to some data using a likeness factor
        /// </summary>
        /// <param name="element"></param>
        /// <param name="likeness">0-100</param>
        /// <returns></returns>
        int CompareTo(object data, int likeness);

        void Scale(double p, double p_2);
    }

    /// <summary>
    /// abstract base for all elements 
    /// 
    /// 
    /// 
    /// </summary>
    public abstract class BaseElement : IElement
    {
        private int m_Id = ElementIDGenerator.New();
        protected List<IElement> m_Children = null;
        protected Rectangle m_Region = Rectangle.Empty;
        protected Rectangle m_ContentRegion = Rectangle.Empty;
        protected MapObject m_MapObject = null;

        abstract protected ElementType GetElementType();

        public ElementType ElementType
        {
            get { return GetElementType(); }
        }

        public int Confidence { get; set; }
        public Bitmap Image { get; set; }

        public Rectangle Region
        {
            get
            {
                return m_Region;
            }
            set
            {
                m_Region = value;
                m_ContentRegion = Rectangle.Empty;
                return;
            }
        }

        public Rectangle ContentRegion
        {
            get
            {
                if (m_ContentRegion == Rectangle.Empty && m_Region != Rectangle.Empty)
                {
                    m_ContentRegion = m_Region;
                    if (ChildCount > 0)
                        foreach (IElement e in m_Children)
                            m_ContentRegion = Rectangle.Union(m_ContentRegion, e.Region);
                }
                return m_ContentRegion;
            }
        }

        public IEnumerable<IElement> Children
        {
            get
            {
                if (m_Children == null) m_Children = new List<IElement>();
                return m_Children;
            }
        }

        public int ChildCount
        {
            get
            {
                if (m_Children == null) return 0;
                else return m_Children.Count;
            }
        }

        public int ID
        {
            get
            {
                return m_Id;
            }
        }

        public object Tag
        {
            get; set;
        }

        public MapObject MapObject
        {
            get { return m_MapObject; }
        }

        public void AddChild(IElement e)
        {
            if (e == null) return;
            if (m_Children == null) m_Children = new List<IElement>();
            m_Children.Add(e);
            m_ContentRegion = Rectangle.Empty;
        }
        public void RemoveChild(IElement e)
        {
            if (e == null || m_Children == null) return;
            m_Children.Remove(e);
            m_ContentRegion = Rectangle.Empty;
        }

        public void AddChild(IEnumerable<IElement> list)
        {
            if (list == null) return;
            m_ContentRegion = Rectangle.Empty;
            if (m_Children == null) m_Children = new List<IElement>();
            foreach (IElement e in list) m_Children.Add(e);
        }


        public BaseElement(MapObject baseobject)
        {
            m_Region = baseobject.Region.Rectangle;
            m_MapObject = baseobject;
        }

        public BaseElement(Rectangle rc)
        {
            m_Region = rc;
            m_MapObject = null;
        }

        public void Scale(double x, double y)
        {
            Region = new Rectangle(
             (int)(Region.Left * x),
             (int)(Region.Top * y),
             (int)(Region.Width * x),
             (int)(Region.Height * y));
            m_ContentRegion = Rectangle.Empty;
        }


        #region compare to 
        virtual public int CompareTo(IElement element, int likeness)
        {
            return 0;
        }

        virtual public int CompareTo(object data, int likeness)
        {
            return 0;
        }
        #endregion

    }



    /// <summary>
    /// generator for unique id;s for elements within session 
    /// </summary>
    internal static class ElementIDGenerator
    {
        static private int m_Id = 0;
        static public int New()
        {
            return Interlocked.Increment(ref m_Id);
        }
    }

}
