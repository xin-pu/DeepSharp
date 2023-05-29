namespace TorchSharpTest.LossTest
{
    public class LossTest : AbstractTest
    {
        public LossTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void Cross()
        {
            var input = torch.randn(3, 5, requires_grad: true);
            var target = torch.empty(3, torch.ScalarType.Int64).randint_like(0, 5);
            var loss = CrossEntropyLoss();
            var c = loss.call(input, target);
            c.backward();
            var array = c.data<float>().ToArray();
            Print(string.Join(",", array));
        }
    }
}