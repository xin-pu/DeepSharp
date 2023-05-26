namespace TorchSharpTest.TorchTests
{
    public class DataSetTest : AbstractTest
    {
        public DataSetTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void DatasetTest()
        {
        }
    }

    public class IrisData
    {
        public int Label;

        public double PetalLength;

        public double PetalWidth;

        public double SepalLength;

        public double SepalWidth;
    }


    public class IrisDataSet : torch.utils.data.Dataset
    {
        public override Dictionary<string, torch.Tensor> GetTensor(long index)
        {
            throw new NotImplementedException();
        }

        public override long Count { get; }
    }
}