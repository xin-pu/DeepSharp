namespace TorchSharpTest
{
    public class AbstractTest
    {
        private readonly ITestOutputHelper _testOutputHelper;

        public AbstractTest(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        protected void writeLint(object? obj)
        {
            _testOutputHelper.WriteLine(obj?.ToString());
        }

        internal void Print(string[] objs)
        {
            foreach (var o in objs) Print(o);
        }

        internal void Print(object obj)
        {
            writeLint(obj);
        }

        protected void Print(Array array)
        {
            writeLint(array);
        }

        /// <summary>
        ///     Todo optimize print Tensor
        /// </summary>
        /// <param name="tensor"></param>
        internal void Print(torch.Tensor tensor)
        {
            writeLint(tensor);
        }

        /// <summary>
        ///     Todo optimize print Tensor
        /// </summary>
        /// <param name="tensor"></param>
        internal void Print(float tensor)
        {
            writeLint(tensor);
        }
    }
}