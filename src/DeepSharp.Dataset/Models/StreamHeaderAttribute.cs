using DeepSharp.Dataset.Models;

namespace DeepSharp.Dataset.Datasets
{
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field)]
    public class StreamHeaderAttribute : Attribute
    {
        public StreamHeaderRange StreamHeaderRange;

        /// <summary>Maps member to specific field in text file.</summary>
        /// <param name="fieldIndex">The index of the field in the text file.</param>
        public StreamHeaderAttribute(int fieldIndex)
        {
            StreamHeaderRange = new StreamHeaderRange(fieldIndex);
        }

        public StreamHeaderAttribute(int min, int max)
        {
            StreamHeaderRange = new StreamHeaderRange(min, max);
        }
    }
}