using DeepSharp.Dataset;
using TorchSharpTest.SampleDataset;

namespace TorchSharpTest.DemoTest
{
    public class IrisTest : AbstractTest
    {
        public IrisTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }


        private torch.Device device => new(DeviceType.CPU);
        public string SaveFile => "Iris.txt";


        /// <summary>
        ///     CrossEntropyLoss
        /// </summary>
        [Fact]
        public async void Train()
        {
            var dataset = new Dataset<IrisOneHot>(@"..\..\..\..\..\Resources\iris-train.txt");
            var dataConfig = new DataLoaderConfig { BatchSize = 8, Device = device };
            var dataloader = new DataLoader<IrisOneHot>(dataset, dataConfig);


            var net = new DemoNet(4, 3).to(device);

            var optimizer = torch.optim.Adam(net.parameters());
            var crossEntropyLoss = CrossEntropyLoss();
            foreach (var epoch in Enumerable.Range(0, 500))
            {
                var lossEpoch = new List<float>();
                await foreach (var datapair in dataloader.GetBatchSample())
                {
                    var (x, y) = (datapair.Features, datapair.Labels);

                    var eval = net.forward(x);
                    var output = crossEntropyLoss.call(eval, y);

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
            net.save(SaveFile);
        }


        [Fact]
        public void Predict()
        {
            using var net = new DemoNet(4, 3);
            {
                net.load(SaveFile);

                var testdata = new IrisOneHot
                {
                    SepalLength = 5.0f,
                    SepalWidth = 3.6f,
                    PetalLength = 1.4f,
                    PetalWidth = 0.2f
                };
                var y = net.forward(testdata.GetFeatures().unsqueeze(0));
                var yy = Softmax(1).call(y);
                var res = yy.argmax().item<long>();

                Print(string.Join(",", yy.data<float>().ToArray()));
                Print(res.ToString());
            }
        }
    }
}