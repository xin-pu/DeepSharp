using DeepSharp.RL.Agents;
using DeepSharp.Utility;

namespace TorchSharpTest.RLTest.ModelTest
{
    public class VTableTest : AbstractTest
    {
        public VTableTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void CreateValueTableTest1()
        {
            var vt = new VTable();
            var state = torch.tensor(new[] {0, 0, 1});
            vt[state] = 3f;
            Print(vt[state]);
            var state2 = torch.tensor(new[] {0, 1, 1});
            Print(vt[state2]);
        }

        [Fact]
        public void CreateValueTableTest2()
        {
            var state1 = torch.tensor(new[] {0, 0, 1});
            var state2 = torch.tensor(new[] {0, 0, 1});
            var arr = new[] {state1, state2};
            var p = arr.Distinct(new TensorEqualityCompare()).ToList();
            Print(p.Count);
        }
    }
}
