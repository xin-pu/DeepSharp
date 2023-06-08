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


        /// <summary>
        ///     Todo optimize print Tensor
        /// </summary>
        /// <param name="tensor"></param>
        internal void Print(torch.Tensor tensor)
        {
            writeLint(tensor.ToString(torch.numpy));
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