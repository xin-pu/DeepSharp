using DeepSharp.RL.Agents;
using FluentAssertions;

namespace TorchSharpTest.RLTest
{
    public class DQNTest : AbstractTest
    {
        public DQNTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }


        [Fact]
        public void TestDQNModel()
        {
            var net = new DQNNet(new long[] {1, 416, 416}, 3, DeviceType.CPU);

            var c = net.children();
            foreach (var module in c) Print(module.GetName());

            var input = torch.randn(1, 1, 416, 416);
            var res = net.forward(input);
            Print(res);

            res.shape.Should().BeEquivalentTo(new long[] {1, 3});
        }
    }
}
