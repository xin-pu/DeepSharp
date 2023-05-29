namespace DeepSharp.Utility.Operations
{
    public class OpMat
    {
        /// <summary>
        ///     Get Dims os Mat
        /// </summary>
        /// <param name="mat"></param>
        /// <returns></returns>
        public static long[] GetDims(Mat mat)
        {
            mat.GetArray(out float[] d);
            var dims = Enumerable.Range(0, mat.Dims)
                .Select(a => (long) mat.Size(a))
                .ToArray();
            return dims;
        }
    }
}