using TorchSharp.Modules;

namespace TorchSharpTest.TorchTests
{
    public class ModuleTest : AbstractTest
    {
        public ModuleTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        private torch.Device device => new(DeviceType.CUDA);

        [Fact]
        public void LinearTest()
        {
            var linear = Linear(4, 5, device: device);
            var x = torch.randn(3, 5, 4, device: device);
            var y = linear.forward(x);
            Print(y);
        }


        [Fact]
        public void NetTest()
        {
            var x = torch.zeros(3, 4).to(device);

            var net = new Net(4, 5, 3).to(device);
            var y = net.Forward(x);

            Print(y);
        }


        [Fact]
        public void TrainTest()
        {
            var x = torch.randn(64, 100).to(device);
            var y = torch.randn(64, 3).to(device);
            var net = new Net(100, 5, 3).to(device);
            var optimizer = torch.optim.Adam(net.parameters());

            for (var i = 0; i < 10; i++)
            {
                var eval = net.Forward(x);
                var output = functional.mse_loss(eval, y, Reduction.Sum);

                optimizer.zero_grad();

                output.backward();

                optimizer.step();

                var loss = output.item<float>();
                Print($"epoch:\t{i:D5}\tLoss:\t{loss}");
            }
        }
    }

    public class Net : Module
    {
        public Net(int obsSize, int hiddenSize, int actionNum)
            : base("Net")
        {
            Sequential = Sequential(
                Linear(obsSize, hiddenSize).to(DeviceType.CUDA),
                ReLU().to(DeviceType.CUDA),
                Linear(hiddenSize, actionNum).to(DeviceType.CUDA));
        }


        public Sequential Sequential { set; get; }

        public override IEnumerable<Parameter> parameters(bool recurse = true)
        {
            return Sequential.parameters();
        }

        public torch.Tensor Forward(torch.Tensor x)
        {
            return Sequential.forward(x);
        }
    }
}