using DeepSharp.RL.Environs;
using FluentAssertions;

namespace TorchSharpTest.RLTest.EnvironTest
{
    public class FrozenLakeTest : AbstractTest
    {
        public DeviceType DeviceType = DeviceType.CUDA;

        public FrozenLakeTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void FrozenLakeCreateTest1()
        {
            var frozenlake = new Frozenlake(deviceType: DeviceType);
            Print(frozenlake);
        }

        [Fact]
        public void FrozenLakeCreate2Test()
        {
            var frozenlake = new Frozenlake(deviceType: DeviceType);
            frozenlake.SetPlayID(15);
            Print(frozenlake);
            frozenlake.IsComplete(1).Should().BeTrue();
        }


        [Fact]
        public void FrozenLakeTestMove()
        {
            var frozenlake = new Frozenlake();
            var testEpoch = 100;
            var count = 0;
            var countL = 0;
            var countR = 0;
            foreach (var i in Enumerable.Range(0, testEpoch))
            {
                frozenlake.SetPlayID(1);
                frozenlake.Step(new Act(torch.from_array(new[] {1})), 1);
                if (frozenlake.PlayID == 5) count++;
                if (frozenlake.PlayID == 0) countL++;
                if (frozenlake.PlayID == 2) countR++;
            }

            var probTarget = count * 1f / testEpoch;
            var probLeft = countL * 1f / testEpoch;
            var probRight = countR * 1f / testEpoch;

            Print($"{probTarget:P2}\t{probLeft:P2}\t{probRight:P2}");
        }
    }
}