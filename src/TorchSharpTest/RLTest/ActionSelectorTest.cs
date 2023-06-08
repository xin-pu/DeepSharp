using DeepSharp.RL.ActionSelectors;
using FluentAssertions;

namespace TorchSharpTest.RLTest
{
    public class ActionSelectorTest : AbstractTest
    {
        public ActionSelectorTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void ArgmaxActionSelectorTest()
        {
            var input = torch.from_array(new double[,] {{1, 2, 3}, {1, -1, 0}});
            var res = new ArgmaxActionSelector().Select(input);
            res.Equals(torch.tensor(new long[] {2, 0})).Should().BeTrue();
        }

        [Fact]
        public void ProbabilityActionSelectorTest()
        {
            var input = torch.from_array(new[,] {{1f, 0, 0}, {0, 1f, 0}});
            var res = new ProbabilityActionSelector().Select(input);
            res.Equals(torch.from_array(new long[] {0, 1})).Should().BeTrue();
        }
    }
}