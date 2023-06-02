using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;

namespace TorchSharpTest.RLTest
{
    public class FrozenLakeTest : AbstractTest
    {
        public FrozenLakeTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void FrozenLakeCreateTest()
        {
            var pro = new Frozenlake();
            Print(pro);
        }

        [Fact]
        public void FrozenLakeCreate2Test()
        {
            var pro = new Frozenlake
            {
                PlayID = 15
            };
            Print(pro);
        }


        [Fact]
        public void Main()
        {
            var epoch = 5000;
            var episodesEachBatch = 100;

            /// Step 1 Create a 4-Armed Bandit
            var forFrozenLake = new Frozenlake();
            Print(forFrozenLake);

            /// Step 2 Create AgentCrossEntropy with 0.7f percentElite as default
            var agent = new AgentCrossEntropyExt(forFrozenLake);

            /// Step 3 Learn and Optimize
            foreach (var i in Enumerable.Range(0, epoch))
            {
                var batch = forFrozenLake.GetMultiEpisodes(agent, episodesEachBatch);
                var success = batch.Count(a => a.SumReward.Value > 0) * 1f / batch.Length;
                var eliteOars = agent.GetElite(batch); /// Get eliteOars 

                /// Agent Learn by elite observation & action
                var loss = agent.Learn(eliteOars);
                var rewardMean = batch.Select(a => a.SumReward.Value).Sum();

                Print($"Epoch:{i:D4}\t:\t{success:p2}\tReward:{rewardMean:F4}\tLoss:{loss:F4}");
                if (success > 0.75)
                    break;
            }
        }
    }
}