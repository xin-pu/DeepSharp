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
            var deviceType = DeviceType.CPU;
            var scalarType = torch.ScalarType.Float32;

            var net = new DQNNet(new long[] {1, 416, 416}, 3,
                scalarType,
                deviceType);

            var c = net.children();
            foreach (var module in c) Print(module.GetName());

            var input = torch.randn(1, 1, 416, 416, scalarType, new torch.Device(deviceType));
            var res = net.forward(input);
            Print(res);

            res.shape.Should().BeEquivalentTo(new long[] {1, 3});
        }
    }
}
