using TorchSharpTest.DemoTest;
using static TorchSharp.torch;

namespace TorchSharpTest.TorchTests
{
    public class ModuleTest : AbstractTest
    {
        public ModuleTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        private Device device => new(DeviceType.CUDA);
        private string savePath => "test.txt";

        [Fact]
        public void LinearTest()
        {
            var linear = Linear(4, 5, device: device);
            var x = randn(3, 5, 4, device: device);
            var y = linear.forward(x);
            Print(y);
        }


        [Fact]
        public void NetTest()
        {
            var x = zeros(3, 4).to(device);

            var net = new DemoNet(4, 3).to(device);
            var y = net.forward(x);

            Print(y);
        }

        [Fact]
        public void NetSaveTest()
        {
            if (File.Exists(savePath)) File.Delete(savePath);
            var net = new DemoNet(4, 3);
            net.save(savePath);

            var a = from_array(new float[] {1, 2, 3, 4});
            var c = net.forward(a);
            var str = string.Join(",", c.data<float>().ToArray());
            Print(str);

            c = net.forward(a);
            str = string.Join(",", c.data<float>().ToArray());
            Print(str);
        }

        [Fact]
        public void NetLoadTest()
        {
            var net = new DemoNet(4, 3);
            net.load(savePath);
            var a = from_array(new float[] {1, 2, 3, 4});
            var c = net.forward(a);
            var str = string.Join(",", c.data<float>().ToArray());
            Print(str);
        }
    }
}