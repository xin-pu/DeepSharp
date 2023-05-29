namespace DeepSharp.Dataset.Models
{
    /// <summary>
    ///     Specifies the range of indices of input columns that should be mapped to an output column.
    /// </summary>
    public sealed class StreamHeaderRange
    {
        /// <summary>
        ///     Whether this range includes only other indices not specified.
        /// </summary>
        public bool AllOther;

        /// <summary>
        ///     Whether this range extends to the end of the line, but should be a fixed number of items.
        ///     If <see cref="Max" /> is specified, the fields <see cref="AutoEnd" /> and <see cref="VariableEnd" /> are ignored.
        /// </summary>
        public bool AutoEnd;

        /// <summary>
        ///     Force scalar columns to be treated as vectors of length one.
        /// </summary>
        public bool ForceVector;

        /// <summary>
        ///     The maximum index of the column, inclusive. If <see langword="null" />
        ///     indicates that the <see cref="TextLoader" /> should auto-detect the length
        ///     of the lines, and read until the end.
        ///     If max is specified, the fields <see cref="AutoEnd" /> and <see cref="VariableEnd" /> are ignored.
        /// </summary>
        public int Max;

        /// <summary>
        ///     The minimum index of the column, inclusive.
        /// </summary>
        public int Min;

        /// <summary>
        ///     Whether this range extends to the end of the line, which can vary from line to line.
        ///     If <see cref="Max" /> is specified, the fields <see cref="AutoEnd" /> and <see cref="VariableEnd" /> are ignored.
        ///     If <see cref="AutoEnd" /> is <see langword="true" />, then <see cref="VariableEnd" /> is ignored.
        /// </summary>
        public bool VariableEnd;

        public StreamHeaderRange()
        {
        }

        /// <summary>
        ///     A range representing a single value. Will result in a scalar column.
        /// </summary>
        /// <param name="index">The index of the field of the text file to read.</param>
        public StreamHeaderRange(int index)
        {
            index.Should().BeGreaterThanOrEqualTo(0, "Must be non-negative");
            Min = index;
            Max = index;
        }

        /// <summary>
        ///     A range representing a set of values. Will result in a vector column.
        /// </summary>
        /// <param name="min">The minimum inclusive index of the column.</param>
        /// <param name="max">
        ///     The maximum-inclusive index of the column. If <c>null</c>
        ///     indicates that the <see cref="TextLoader" /> should auto-detect the length
        ///     of the lines, and read until the end.
        /// </param>
        public StreamHeaderRange(int min, int max)
        {
            min.Should().BeGreaterThanOrEqualTo(0, "Must be non-negative");
            max.Should().BeGreaterOrEqualTo(min, "If specified, must be greater than or equal to " + nameof(min));

            Min = min;
            Max = max;
            // Note that without the following being set, in the case where there is a single range
            // where Min == Max, the result will not be a vector valued but a scalar column.
            ForceVector = true;
        }

        internal static StreamHeaderRange Parse(string str)
        {
            str.Should().NotBeNullOrEmpty();
            var res = new StreamHeaderRange();
            if (res.TryParse(str)) return res;

            return null;
        }

        private bool TryParse(string str)
        {
            str.Should().NotBeNullOrEmpty();

            var ich = str.IndexOfAny(new[] {'-', '~'});
            if (ich < 0)
            {
                // No "-" or "~". Single integer.
                if (!int.TryParse(str, out Min)) return false;

                Max = Min;
                return true;
            }

            AllOther = str[ich] == '~';
            ForceVector = true;

            if (ich == 0)
            {
                if (!AllOther) return false;

                Min = 0;
            }
            else if (!int.TryParse(str.Substring(0, ich), out Min))
            {
                return false;
            }

            var rest = str.Substring(ich + 1);
            if (string.IsNullOrEmpty(rest) || rest == "*")
            {
                AutoEnd = true;
                return true;
            }

            if (rest == "**")
            {
                VariableEnd = true;
                return true;
            }

            int tmp;
            if (!int.TryParse(rest, out tmp)) return false;

            Max = tmp;
            return true;
        }

        internal bool TryUnparse(StringBuilder sb)
        {
            sb.Should().NotBeNull();
            var dash = AllOther ? '~' : '-';
            if (Min < 0) return false;

            sb.Append(Min);
            if (Max != null)
            {
                if (Max != Min || ForceVector || AllOther) sb.Append(dash).Append(Max);
            }
            else if (AutoEnd)
            {
                sb.Append(dash).Append("*");
            }
            else if (VariableEnd)
            {
                sb.Append(dash).Append("**");
            }

            return true;
        }
    }
}