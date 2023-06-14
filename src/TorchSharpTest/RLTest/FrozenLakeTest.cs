using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;

namespace TorchSharpTest.RLTest
{
    public class FrozenLakeTest : AbstractTest
    {
        public DeviceType DeviceType = DeviceType.CUDA;

        public FrozenLakeTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void FrozenLakeCreateTest1()
        {
            var pro = new Frozenlake(deviceType: DeviceType);
            Print(pro);
        }

        [Fact]
        public void FrozenLakeCreate2Test()
        {
            var pro = new Frozenlake(deviceType: DeviceType) {PlayID = 15};
            Print(pro);
            var res = pro.IsComplete(1);
            Print($"{res}");
        }


        [Fact]
        public void AgentCrossEntropyMain()
        {
            var epoch = 100;
            var episodesEachBatch = 100;

            /// Step 1 Create a 4-Armed Bandit
            var forFrozenLake = new Frozenlake(deviceType: DeviceType.CPU) {Gamma = 0.90f};
            Print(forFrozenLake);

            /// Step 2 Create AgentCrossEntropy with 0.7f percentElite as default
            var agent = new AgentCrossEntropyExt(forFrozenLake)
            {
                MemsEliteLength = 30
            };

            /// Step 3 Learn and Optimize
            foreach (var i in Enumerable.Range(0, epoch))
            {
                var batch = agent.PlayEpisode(episodesEachBatch);
                var success = batch.Count(a => a.SumReward.Value > 0);

                var eliteOars = agent.GetElite(batch); /// Get eliteOars 

                /// Agent Learn by elite observation & action
                var loss = agent.Learn(eliteOars);
                var rewardMean = batch.Select(a => a.SumReward.Value).Sum();

                Print($"Epoch:{i:D4}\t:\t{success}\tReward:{rewardMean:F4}\tLoss:{loss:F4}");
            }
        }

        [Fact]
        public void QLearningRunRandom()
        {
            /// Step 1 Create a 4-Armed Bandit
            var kArmedBandit = new Frozenlake(deviceType: DeviceType) {Gamma = 0.90f};
            Print(kArmedBandit);

            /// Step 2 Create AgentCrossEntropy with 0.7f percentElite as default
            var agent = new AgentQLearning(kArmedBandit);
            agent.RunRandom(kArmedBandit, 100);
            agent.ValueIteration();
        }

        [Fact]
        public void QLearningMain()
        {
            /// Step 1 Create a 4-Armed Bandit
            var frozenLake = new Frozenlake(deviceType: DeviceType.CPU) {Gamma = 0.95f};

            /// Step 2 Create AgentCrossEntropy with 0.7f percentElite as default
            var agent = new AgentQLearning(frozenLake);
            Print(frozenLake);

            var i = 0;
            var testEpisode = 10;
            var bestReward = 0f;
            while (true)
            {
                agent.RunRandom(frozenLake, 100);
                agent.ValueIteration();

                var episodes = Enumerable.Range(0, testEpisode)
                    .Select(i1 => agent.PlayEpisode())
                    .ToList();
                var sum = episodes.Average(a => a.SumReward.Value);
                bestReward = new[] {bestReward, sum}.Max();
                Print($"{agent} Play:{++i:D3}\t reward:{sum:F2}");
                if (i > 100)
                    break;
            }

            frozenLake.CallBack = s => { Print(frozenLake); };
            agent.PlayEpisode();
        }
    }
}