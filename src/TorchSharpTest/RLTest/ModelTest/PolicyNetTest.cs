using DeepSharp.RL.Agents;

namespace TorchSharpTest.RLTest.ModelTest
{
    public class PolicyNetTest : AbstractTest
    {
        public PolicyNetTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void TestNet()
        {
            var net = new PGN(3, 128, 2, DeviceType.CPU);
            var res = net.forward(torch.from_array(new float[,] {{1, 2, 3}}));
            Print(res);
        }
    }
}