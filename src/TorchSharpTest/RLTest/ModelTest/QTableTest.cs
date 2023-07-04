using DeepSharp.RL.Agents;
using FluentAssertions;

namespace TorchSharpTest.RLTest.ModelTest
{
    public class QTableTest : AbstractTest
    {
        public QTableTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void TestTransitKeyOperator()
        {
            var state1 = torch.tensor(new[] {0, 0, 1});
            var action1 = torch.tensor(new[] {0, 1, 0});
            var key1 = new TransitKey(state1, action1);


            var state2 = torch.tensor(new[] {0, 0, 1});
            var action2 = torch.tensor(new[] {0, 1, 0});
            var key2 = new TransitKey(state2, action2);

            (key2 == key1).Should().BeTrue();
        }

        [Fact]
        public void TestTransitKeyDict()
        {
            var state = torch.tensor(new[] {0, 0, 1});
            var action = torch.tensor(new[] {0, 1, 0});
            var key = new TransitKey(state, action);
            var returnDict = new Dictionary<TransitKey, float> {[key] = 2};

            var stateTest = torch.tensor(new[] {0, 0, 1});
            var actionTest = torch.tensor(new[] {0, 1, 0});
            var keyTest = new TransitKey(stateTest, actionTest);
            var res = returnDict[keyTest];
            res.Should().Be(2);
        }


        [Fact]
        public void CreateValueTableTest1()
        {
            var vt = new QTable();
            var state = torch.tensor(new[] {0, 0, 1});
            var action = torch.tensor(new[] {0, 1, 0});
            var tr = new TransitKey(state, action);
            vt[tr] = 3f;
            Print(vt[tr]);
            var state2 = torch.tensor(new[] {0, 1, 1});
            Print(vt[state2, action]);
        }

        [Fact]
        public void CreateValueTableTest2()
        {
            var vt = new QTable();
            var state = torch.tensor(new[] {0, 0, 1});
            var action = torch.tensor(new[] {0, 1, 0});
            vt[state, action] = 3f;
            Print(vt[state, action]);
            var state2 = torch.tensor(new[] {0, 1, 1});
            Print(vt[state2, action]);
        }
    }
}