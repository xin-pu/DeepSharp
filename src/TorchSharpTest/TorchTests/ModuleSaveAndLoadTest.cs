using TorchSharp.Modules;

namespace TorchSharpTest.TorchTests
{
    public class ModuleSaveAndLoadTest : AbstractTest
    {
        public ModuleSaveAndLoadTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        public Sequential GetNet(int obsSize = 4, int actionNum = 3)
        {
            return Sequential(
                Linear(obsSize, 10).to(DeviceType.CUDA),
                Linear(10, actionNum).to(DeviceType.CUDA));
        }

        public string Location => "Test.ts";

        [Fact]
        public void TestSave()
        {
            if (File.Exists(Location)) File.Delete(Location);
            using var conv = Sequential(
                Conv2d(100, 10, 5),
                Linear(100, 10));
            var params0 = conv.parameters();
            conv.save(Location);
        }

        [Fact]
        public void TestLoad()
        {
            using var loaded =
                Sequential(
                    Conv2d(100, 10, 5),
                    Linear(100, 10));
            loaded.load(Location);
            var params1 = loaded.parameters();
        }
    }
}
