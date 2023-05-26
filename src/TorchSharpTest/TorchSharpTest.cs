using Xunit.Abstractions;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharpTest
{
    public class TorchSharpTest : AbstractTest
    {
        public TorchSharpTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }


        [Fact]
        public void Test1()
        {
            var lin1 = Linear(4, 5);

            var x = randn(3, 5, 4);

            var y = lin1.forward(x);
            Print(y);
        }

        [Fact]
        public void Test2()
        {
        }
    }
}