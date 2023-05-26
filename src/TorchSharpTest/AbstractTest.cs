using Xunit.Abstractions;

namespace TorchSharpTest
{
    public class AbstractTest
    {
        private readonly ITestOutputHelper _testOutputHelper;

        public AbstractTest(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        internal void Print(object obj)
        {
            _testOutputHelper.WriteLine(obj.ToString());
        }
    }
}
