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

        public string SaveFile => "StatModel.ts";

        [Fact]
        public async void TrainTest()
        {
            var dataset = new Dataset<IrisData>(@"F:\Iris\iris-train.txt");
            var dataConfig = new DataLoaderConfig {BatchSize = 8};
            var dataloader = new DataLoader<IrisData>(dataset, dataConfig);


            using var net = Sequential(
                Linear(4, 10),
                Linear(10, 3));
            {
                var r = net.to(device);
                var optimizer = torch.optim.Adam(r.parameters());
                var bceWithLogitsLoss = CrossEntropyLoss();
                foreach (var epoch in Enumerable.Range(0, 500))
                {
                    var lossEpoch = new List<float>();
                    await foreach (var datapair in dataloader.GetBatchSample())
                    {
                        var (x, y) = (datapair.Features, datapair.Labels);

                        var eval = r.forward(x);
                        var y_resharp = y.squeeze(-1);
                        var output = bceWithLogitsLoss.call(eval, y_resharp);

                        optimizer.zero_grad();
                        output.backward();
                        optimizer.step();

                        var loss = output.item<float>();
                        lossEpoch.Add(loss);
                    }

                    var t = lossEpoch.Average();
                    Print($"epoch:\t{epoch:D5}\tLoss:\t{t:F4}");
                }

                if (File.Exists(SaveFile)) File.Delete(SaveFile);

                r.save(SaveFile);
            }
        }

        [Fact]
        public void Predict()
        {
            using var net = Sequential(
                Linear(4, 10),
                Linear(10, 3));
            {
                net.load("StatModel.ts");

                var testdata = new IrisData
                {
                    SepalLength = 4.6f,
                    SepalWidth = 3.1f,
                    PetalLength = 1.5f,
                    PetalWidth = 0.2f
                };
                var y = net.forward(testdata.GetFeatures().unsqueeze(0));

                var arr = y.data<float>().ToArray();
                var res = y.argmax().item<long>();

                Print(string.Join(",", arr));
                Print(res.ToString());
            }
        }

        public Sequential GetNet(int obsSize = 4, int actionNum = 3)
        {
            return Sequential(
                Linear(obsSize, 10).to(DeviceType.CUDA),
                Linear(10, actionNum).to(DeviceType.CUDA));
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