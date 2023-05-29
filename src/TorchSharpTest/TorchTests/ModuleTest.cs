using DeepSharp.Dataset;
using TorchSharp.Modules;
using TorchSharpTest.SampleDataset;

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

            var net = new Net(4, 3).to(device);
            var y = net.Forward(x);

            Print(y);
        }


        [Fact]
        public void TrainTest()
        {
            var x = torch.randn(64, 100).to(device);
            var y = torch.randn(64, 3).to(device);
            var net = new Net(100, 3).to(device);
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

        [Fact]
        public async void TrainTest2()
        {
            var dataset = new Dataset<IrisData>(@"F:\Iris\iris-train.txt");
            var dataConfig = new DataLoaderConfig {BatchSize = 8};
            var dataloader = new DataLoader<IrisData>(dataset, dataConfig);

            var net = new Net(4, 3).to(device);
            var optimizer = torch.optim.SGD(net.parameters(), 1E-3);


            var bceWithLogitsLoss = BCEWithLogitsLoss();
            foreach (var epoch in Enumerable.Range(0, 100))
            {
                var lossEpoch = new List<float>();
                await foreach (var datapair in dataloader.GetBatchSample())
                {
                    var (x, y) = (datapair.Features, datapair.Labels);

                    var eval = net.Forward(x);
                    var output = bceWithLogitsLoss.call(eval, y);

                    optimizer.zero_grad();
                    output.backward();
                    optimizer.step();

                    var loss = output.item<float>();
                    lossEpoch.Add(loss);
                }

                var t = lossEpoch.Average();
                Print($"epoch:\t{epoch:D5}\tLoss:\t{t:F4}");
            }


            net.save("StatModel.ts");
        }

        [Fact]
        public void LoadModel()
        {
            var net = new Net(4, 3);
            net.load("StatModel.ts");
            var testdata = new IrisData {PetalLength = 5.9f, PetalWidth = 2.1f, SepalLength = 7.1f, SepalWidth = 3.0f};
            var y = net.Forward(testdata.GetFeatures().unsqueeze(0).to(device));
            var z = y.sigmoid().data<float>().ToArray();
            Print(string.Join(",", z));
        }
    }


    public class Net : Module
    {
        public Net(int obsSize, int actionNum)
            : base("Net")
        {
            Sequential = Sequential(
                Linear(obsSize, 10).to(DeviceType.CUDA),
                Linear(10, actionNum).to(DeviceType.CUDA));
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