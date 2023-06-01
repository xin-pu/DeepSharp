namespace DeepSharp.RL.Environs
{
    public class LakeUnit
    {
        /// <summary>
        /// </summary>
        /// <param name="row"></param>
        /// <param name="column"></param>
        public LakeUnit(int row, int column, int index)
        {
            Column = column;
            Row = row;
            Index = index;
            Role = LakeRole.Ice;
        }

        public int Index { set; get; }
        public int Row { set; get; }
        public int Column { set; get; }
        public LakeRole Role { set; get; }

        public override string ToString()
        {
            switch (Role)
            {
                case LakeRole.Ice:
                    return "I";
                case LakeRole.Hole:
                    return "H";
                case LakeRole.Start:
                    return "S";
                case LakeRole.End:
                    return "E";
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
    }
}