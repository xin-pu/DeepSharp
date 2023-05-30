namespace TorchSharpTest.TorchTests
{
    public class SaveLoadTest : AbstractTest
    {
        public SaveLoadTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }


        public string Location => "Test.ts";

        [Fact]
        public void TestSave()
        {
            if (File.Exists(Location)) File.Delete(Location);
            using var conv = Sequential(
                Conv2d(100, 10, 5),
                Linear(100, 10));
            conv.save(Location);
        }

        [Fact]
        public void TestLoad()
        {
            using var loaded = Sequential(
                Conv2d(100, 10, 5),
                Linear(100, 10));
            loaded.load(Location);
        }
    }
}