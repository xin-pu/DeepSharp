


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
            var device = new torch.Device(DeviceType.CUDA);

            var linear = Linear(4, 5, device: device);
            var x = torch.randn(3, 5, 4, device: device);
            var y = linear.forward(x);
            Print(y);
        }

        [Fact]
        public void Test2()
        {
            var device = new torch.Device(DeviceType.CUDA);
            var tensor = torch.randn(3, 5, 4, device: device);
            var ndArray = tensor.data<float>().ToNDArray();
            Print(ndArray);
        }
    }
}