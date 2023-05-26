namespace TorchSharpTest
{
    public class TensorTest : AbstractTest
    {
        public TensorTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void CreateRandTensor()
        {
            var device = new torch.Device(DeviceType.CUDA);
            var tensor = torch.randn(3, 5, 4, device: device);
            Print(tensor.ToString());
        }

        [Fact]
        public void CreateArrayTensor()
        {
            var tensor = torch.from_array(new float[] {1, 2}).to(DeviceType.CUDA);
            Print(tensor);
        }

        [Fact]
        public void CreateOnesTensor()
        {
            var tensor = torch.ones(2, 3).to(DeviceType.CUDA);
            Print(tensor);
        }
    }
}